import asyncio
import json
import time
from datetime import datetime
from threading import Thread
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import config
import state
import model_cache as mcache
from loader import load_model, extract_text_from_pipeline_result
from gguf import build_llama_cpp_prompt
from multimodal import load_multimodal_model, run_multimodal_inference, stream_multimodal_response
from schemas import ChatRequest, MultimodalChatRequest
from utils import with_timeout, gpu_cleanup_fn

router = APIRouter()


# ---------------------------------------------------------------------------
# Streaming generators
# ---------------------------------------------------------------------------

async def _stream_chat(model_name: str, messages: List[Dict[str, str]],
                       max_tokens: int, temperature: float):
    if state.current_model_name != model_name:
        try:
            load_model(model_name)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
            return

    if state.current_model is None:
        yield f"data: {json.dumps({'error': 'No model loaded', 'done': True})}\n\n"
        return

    backend = state.model_meta.get(state.current_model_name, {}).get("backend")
    prompt = _build_prompt(messages)

    if backend == "vllm":
        async for chunk in _stream_vllm(prompt, max_tokens, temperature):
            yield chunk
        return

    if backend and backend.startswith("transformers"):
        async for chunk in _stream_transformers(messages, prompt, max_tokens, temperature):
            yield chunk
        return

    if backend == "gguf_llama_cpp":
        async for chunk in _stream_gguf(messages, max_tokens, temperature):
            yield chunk
        return

    yield f"data: {json.dumps({'error': f'Streaming not supported for backend: {backend}', 'done': True})}\n\n"


async def _stream_vllm(prompt: str, max_tokens: int, temperature: float):
    if config.SamplingParams is None:
        yield f"data: {json.dumps({'error': 'SamplingParams unavailable', 'done': True})}\n\n"
        return
    try:
        params = config.SamplingParams(temperature=temperature, max_tokens=max_tokens)
        for output in state.current_model.generate([prompt], params, use_tqdm=False):
            delta = output.outputs[0].text
            yield f"data: {json.dumps({'delta': {'content': delta}, 'done': False})}\n\n"
        yield f"data: {json.dumps({'delta': {}, 'done': True})}\n\n"
    except Exception as e:
        config.log.exception("vLLM streaming failed")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


async def _stream_transformers(messages, prompt: str, max_tokens: int, temperature: float):
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        yield f"data: {json.dumps({'error': 'TextIteratorStreamer unavailable', 'done': True})}\n\n"
        return

    pipe = state.current_model
    tokenizer = getattr(pipe, "tokenizer", None)
    model_obj = getattr(pipe, "model", None)

    if tokenizer is None or model_obj is None:
        yield f"data: {json.dumps({'error': 'Pipeline lacks tokenizer/model for streaming', 'done': True})}\n\n"
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate():
        try:
            try:
                pipe(messages, max_new_tokens=max_tokens, temperature=temperature,
                     do_sample=temperature > 0.0, streamer=streamer)
            except (TypeError, AttributeError, ValueError):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                ids = inputs["input_ids"]
                try:
                    ids = ids.to(next(model_obj.parameters()).device)
                except Exception:
                    pass
                gk = {"max_new_tokens": max_tokens, "streamer": streamer}
                gk.update({"do_sample": True, "temperature": temperature} if temperature > 0.0
                           else {"do_sample": False})
                model_obj.generate(input_ids=ids, **gk)
        except Exception:
            config.log.exception("Error in transformers generation thread")

    thread = Thread(target=_generate)
    thread.start()
    try:
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'delta': {'content': text}, 'done': False})}\n\n"
    except Exception as e:
        config.log.exception("TextIteratorStreamer error")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        thread.join(timeout=5)
        return
    thread.join(timeout=config.STREAMING_THREAD_TIMEOUT)
    yield f"data: {json.dumps({'delta': {}, 'done': True})}\n\n"


async def _stream_gguf(messages, max_tokens: int, temperature: float):
    llm = state.current_model
    prompt_text = build_llama_cpp_prompt(messages)
    try:
        for chunk in llm(prompt_text, max_tokens=max_tokens, temperature=temperature,
                         stop=["<|user|>", "<|system|>"], stream=True):
            delta = chunk["choices"][0].get("text", "")
            if delta:
                yield f"data: {json.dumps({'delta': {'content': delta}, 'done': False})}\n\n"
        yield f"data: {json.dumps({'delta': {}, 'done': True})}\n\n"
    except Exception as e:
        config.log.exception("GGUF streaming failed")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------

def _build_prompt(messages) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content", "") if isinstance(msg, dict) else (
            msg.content if isinstance(msg.content, str) else "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def _make_response(model_name: str, text: str, gen_duration: int,
                   prompt_tokens=None, gen_tokens=None) -> dict:
    load_duration = state.model_meta.get(state.current_model_name, {}).get("load_duration_ns", 0)
    return {
        "model": model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        "total_duration": gen_duration + load_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": None,
        "eval_count": gen_tokens,
        "eval_duration": gen_duration,
    }


def generate_response(request: ChatRequest, model_name: str, prompt: str, backend: str) -> dict:
    if backend == "vllm":
        if config.SamplingParams is None:
            raise HTTPException(status_code=500, detail="SamplingParams unavailable. Install vllm.")
        params = config.SamplingParams(temperature=request.temperature, max_tokens=request.max_tokens)
        start = time.time_ns()
        outputs = state.current_model.generate([prompt], params)
        duration = time.time_ns() - start
        text = outputs[0].outputs[0].text.strip() if outputs else ""
        return _make_response(model_name, text, duration)

    if backend and backend.startswith("transformers"):
        if not config.transformers_available:
            raise HTTPException(status_code=500, detail="Transformers not available.")
        pipe = state.current_model
        model_obj = getattr(pipe, "model", None)
        tokenizer = getattr(pipe, "tokenizer", None)
        try:
            start = time.time_ns()
            if model_obj is not None and tokenizer is not None:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
                ids = inputs.get("input_ids")
                mask = inputs.get("attention_mask")
                try:
                    dev = next(model_obj.parameters()).device
                    if ids is not None:
                        ids = ids.to(dev)
                    if mask is not None:
                        mask = mask.to(dev)
                except Exception:
                    pass
                gk: dict = {
                    "max_new_tokens": request.max_tokens,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if request.temperature and request.temperature > 0.0:
                    gk.update({"do_sample": True, "temperature": request.temperature, "top_p": 0.95})
                else:
                    gk["do_sample"] = False
                model_cfg = getattr(model_obj, "config", None)
                if (getattr(model_cfg, "model_type", "") in ("phi3", "phi")
                        or "phi" in model_name.lower()):
                    gk["use_cache"] = False
                try:
                    out_ids = model_obj.generate(input_ids=ids, attention_mask=mask, **gk)
                except Exception as e:
                    if any(k in str(e).lower() for k in ("cache", "seen_tokens", "dynamiccache")):
                        gk["use_cache"] = False
                        out_ids = model_obj.generate(input_ids=ids, attention_mask=mask, **gk)
                    else:
                        raise
                gen_text = tokenizer.batch_decode(out_ids[:, ids.shape[1]:], skip_special_tokens=True)
                result: list = [{"generated_text": gen_text[0] if gen_text else ""}]
            else:
                config.log.warning("Pipeline fallback (no model/tokenizer access)")
                result = pipe(prompt, max_new_tokens=request.max_tokens,
                              temperature=request.temperature, do_sample=request.temperature > 0.0)
            duration = time.time_ns() - start
            text = extract_text_from_pipeline_result(result).strip()
        except Exception as e:
            config.log.exception("Transformers generation failed for %s", model_name)
            raise HTTPException(status_code=500,
                                detail=f"Generation failed: {e}. Try restarting or using a smaller model.")

        p_tok = g_tok = None
        try:
            tok = getattr(pipe, "tokenizer", None)
            if tok:
                p_tok = len(tok(prompt).get("input_ids", []))
                g_tok = len(tok(text).get("input_ids", []))
        except Exception:
            pass
        return _make_response(model_name, text, duration, p_tok, g_tok)

    if backend == "gguf_llama_cpp":
        llm = state.current_model
        prompt_text = build_llama_cpp_prompt(request.messages)
        try:
            start = time.time_ns()
            output = llm(prompt_text, max_tokens=request.max_tokens,
                         temperature=request.temperature, stop=["<|user|>", "<|system|>"])
            duration = time.time_ns() - start
            text = output["choices"][0]["text"].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GGUF generation failed: {e}")
        resp = _make_response(model_name, text, duration)
        resp["eval_count"] = output.get("usage", {}).get("completion_tokens")
        return resp

    raise HTTPException(status_code=500,
                        detail=f"Unknown backend '{backend}'. Supported: vllm, transformers, gguf_llama_cpp.")


async def _chat_non_streaming(request: ChatRequest, model_name: str) -> dict:
    if state.current_model_name != model_name:
        try:
            await asyncio.to_thread(load_model, model_name)
        except Exception as e:
            msg = str(e)
            if "not downloaded" in msg:
                raise HTTPException(status_code=400, detail=msg)
            if "cooldown" in msg:
                raise HTTPException(status_code=429, detail=msg)
            if "Missing dependency" in msg:
                raise HTTPException(status_code=400, detail=msg)
            raise HTTPException(status_code=500, detail=f"Failed to load '{model_name}': {msg}")

    if state.current_model is None:
        raise HTTPException(status_code=500, detail="No model loaded.")

    prompt = _build_prompt(request.messages)
    backend = state.model_meta.get(state.current_model_name, {}).get("backend")
    return await asyncio.to_thread(generate_response, request, model_name, prompt, backend)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    summary="Text chat inference",
    description=(
        "Run a text chat completion against a loaded model. Returns an Ollama-compatible "
        "JSON response with `message.content`, `done`, `total_duration`, etc.\n\n"
        "Set `stream: true` to receive Server-Sent Events instead of a single JSON response."
    ),
    tags=["chat"],
    responses={
        200: {"description": "Successful chat completion."},
        400: {"description": "Bad request (missing model, invalid timeout, etc.)."},
        408: {"description": "Request exceeded the configured timeout."},
        429: {"description": "Model is in cooldown after a recent failed load."},
        500: {"description": "Inference failed (GPU OOM, backend error, etc.)."},
        503: {"description": "Model could not be loaded."},
    },
)
async def chat(request: ChatRequest):
    timeout_s = request.timeout or 120
    if not 1 <= timeout_s <= 600:
        raise HTTPException(status_code=400, detail="Timeout must be 1–600 seconds.")

    if request.stream:
        return StreamingResponse(
            _stream_chat(request.model, request.messages, request.max_tokens, request.temperature),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return await with_timeout(_chat_non_streaming(request, request.model),
                               timeout_s, gpu_cleanup_fn)


@router.post(
    "/chat/multimodal",
    summary="Multimodal chat (text + image/audio/video)",
    description=(
        "Run a multimodal chat completion. Accepts HF-style content blocks "
        "(`[{\"type\":\"image\",\"url\":\"...\"}, {\"type\":\"text\",\"text\":\"...\"}]`) "
        "or the legacy `images`/`audio`/`video` fields per message.\n\n"
        "Falls back to text-only inference when no media is supplied."
    ),
    tags=["multimodal"],
    responses={
        200: {"description": "Successful multimodal completion."},
        400: {"description": "GGUF model id or invalid request."},
        503: {"description": "Model could not be loaded."},
    },
)
async def chat_multimodal(request: MultimodalChatRequest):
    model_name = request.model
    if model_name.lower().endswith(".gguf"):
        raise HTTPException(status_code=400,
                            detail="GGUF not supported for multimodal. Use a HF repo id.")

    timeout_s = max(1, min(request.timeout or 180, 600))
    has_media = any(m.images or m.audio or m.video for m in request.messages)

    if model_name not in state.model_cache:
        try:
            if has_media:
                await asyncio.to_thread(load_multimodal_model, model_name)
            else:
                await asyncio.to_thread(load_model, model_name)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    elif has_media and model_name not in state.mm_processor_cache:
        try:
            await asyncio.to_thread(load_multimodal_model, model_name)
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

    if request.stream:
        if has_media:
            return StreamingResponse(
                stream_multimodal_response(model_name, request.messages,
                                           request.max_tokens, request.temperature),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        text_msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        return StreamingResponse(
            _stream_chat(model_name, text_msgs, request.max_tokens, request.temperature),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    if has_media:
        async def _run():
            reply = await asyncio.to_thread(
                run_multimodal_inference, model_name, request.messages,
                request.max_tokens, request.temperature,
            )
            return {"reply": reply, "model": model_name, "multimodal": True}
        return await with_timeout(_run(), timeout_s, gpu_cleanup_fn)

    def _text_content(m) -> str:
        if isinstance(m.content, str):
            return m.content
        return "\n".join(
            p.get("text", "") for p in m.content
            if isinstance(p, dict) and p.get("type") == "text"
        ) if isinstance(m.content, list) else ""

    text_req = ChatRequest(
        model=model_name,
        messages=[{"role": m.role, "content": _text_content(m)} for m in request.messages],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=False,
        timeout=timeout_s,
    )
    return await with_timeout(_chat_non_streaming(text_req, model_name), timeout_s, gpu_cleanup_fn)
