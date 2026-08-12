import base64
import os
import tempfile
import time
from pathlib import Path
from typing import List

import config
import state
import model_cache as mcache


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------

def decode_media(b64_or_url: str, suffix: str) -> str:
    """Write base64 or URL content to a temp file; return path."""
    if b64_or_url.startswith("http://") or b64_or_url.startswith("https://"):
        import urllib.request
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        urllib.request.urlretrieve(b64_or_url, tmp.name)
        return tmp.name
    if "," in b64_or_url[:80]:
        b64_or_url = b64_or_url.split(",", 1)[1]
    data = base64.b64decode(b64_or_url)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    return tmp.name


def build_mm_inputs(processor, messages):
    from PIL import Image as _PILImage

    pil_images, audio_inputs, video_inputs, tmp_files = [], [], [], []
    conv = []

    for msg in messages:
        content_parts: list = []
        text_parts: List[str] = []

        if isinstance(msg.content, list):
            for part in msg.content:
                if not isinstance(part, dict):
                    continue
                p_type = str(part.get("type", "")).lower()
                if p_type == "image":
                    ref = part.get("url") or part.get("image")
                    if ref:
                        fpath = decode_media(ref, ".jpg")
                        tmp_files.append(fpath)
                        pil_images.append(_PILImage.open(fpath).convert("RGB"))
                        content_parts.append({"type": "image"})
                elif p_type == "audio":
                    ref = part.get("url") or part.get("audio")
                    if ref:
                        fpath = decode_media(ref, ".wav")
                        tmp_files.append(fpath)
                        audio_inputs.append(fpath)
                        content_parts.append({"type": "audio"})
                elif p_type == "video":
                    ref = part.get("url") or part.get("video")
                    if ref:
                        fpath = decode_media(ref, ".mp4")
                        tmp_files.append(fpath)
                        video_inputs.append(fpath)
                        content_parts.append({"type": "video"})
                elif p_type == "text":
                    val = part.get("text")
                    if val:
                        text_parts.append(val)

        if msg.images:
            for img in msg.images:
                fpath = decode_media(img, ".jpg")
                tmp_files.append(fpath)
                pil_images.append(_PILImage.open(fpath).convert("RGB"))
                content_parts.append({"type": "image"})
        if msg.audio:
            fpath = decode_media(msg.audio, ".wav")
            tmp_files.append(fpath)
            audio_inputs.append(fpath)
            content_parts.append({"type": "audio"})
        if msg.video:
            fpath = decode_media(msg.video, ".mp4")
            tmp_files.append(fpath)
            video_inputs.append(fpath)
            content_parts.append({"type": "video"})

        if isinstance(msg.content, str) and msg.content:
            text_parts.append(msg.content)

        content_parts.append({"type": "text", "text": "\n".join(t for t in text_parts if t).strip()})
        conv.append({"role": msg.role, "content": content_parts})

    try:
        prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    except Exception:
        def _text(m) -> str:
            if isinstance(m.content, str):
                return m.content
            return "\n".join(
                p.get("text", "") for p in m.content
                if isinstance(p, dict) and p.get("type") == "text"
            ) if isinstance(m.content, list) else ""
        prompt = "\n".join(f"{m.role}: {_text(m)}" for m in messages) + "\nassistant:"

    proc_kwargs = {"text": prompt, "return_tensors": "pt"}
    if pil_images:
        proc_kwargs["images"] = pil_images
    if audio_inputs:
        try:
            import soundfile as sf
            import numpy as np
            proc_kwargs["audio"] = [sf.read(p)[0].astype(np.float32) for p in audio_inputs]
        except ImportError:
            config.log.warning("[multimodal] soundfile not installed — audio skipped")
    if video_inputs:
        proc_kwargs["videos"] = video_inputs

    inputs = processor(**proc_kwargs)

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return inputs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_multimodal_model(model_name: str) -> None:
    if state.current_model_name == model_name and model_name in state.model_cache:
        return
    if model_name.lower().endswith(".gguf"):
        raise RuntimeError(
            "GGUF is not supported by the Transformers multimodal path. "
            "Use a HuggingFace repo id and /pull first."
        )

    mcache.evict_lru_if_needed()
    config.log.info("[multimodal] Loading: %s", model_name)

    local_path = state.model_meta.get(model_name, {}).get("local_path")
    if local_path is None and config.hf_hub_available and config.snapshot_download is not None:
        try:
            local_path = config.snapshot_download(repo_id=model_name, local_files_only=True)
            state.model_meta[model_name] = {**state.model_meta.get(model_name, {}), "local_path": local_path}
        except Exception:
            pass

    if local_path is None:
        raise RuntimeError(f"Model '{model_name}' not downloaded. POST /pull {{\"model\":\"{model_name}\"}}")

    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch as _torch

        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)

        load_kwargs = dict(trust_remote_code=True, device_map="auto", local_files_only=True)

        try:
            model_size = sum(f.stat().st_size for f in Path(local_path).rglob("*.safetensors")) \
                if Path(local_path).exists() else 0
        except Exception:
            model_size = 0

        if config.bitsandbytes_available and model_size >= config.Q4_THRESHOLD_BYTES:
            from transformers import BitsAndBytesConfig
            config.log.info("[multimodal] 4-bit quant (%.1fGB, layer-split GPU/CPU)", model_size / 1024**3)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=_torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            # Layer-split: cap GPU to leave safety buffer, allow CPU offload
            mm: dict = {}
            if _torch.cuda.is_available():
                mm[0] = config.MAX_GPU_MEMORY
            mm["cpu"] = config.MAX_CPU_MEMORY
            load_kwargs["max_memory"] = mm
        else:
            load_kwargs["torch_dtype"] = _torch.bfloat16

        model = None
        try:
            import transformers as _tf
            mm_cls = getattr(_tf, "AutoModelForMultimodalLM", None)
            if mm_cls is None:
                raise RuntimeError("AutoModelForMultimodalLM not available")
            model = mm_cls.from_pretrained(local_path, **load_kwargs)
            config.log.info("[multimodal] Loaded via AutoModelForMultimodalLM: %s", model_name)
        except Exception:
            config.log.warning("[multimodal] Falling back to AutoModelForCausalLM for %s", model_name)
            model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
        model.eval()

        state.model_cache[model_name] = {"model": model, "tokenizer": processor, "backend": "multimodal"}
        state.mm_processor_cache[model_name] = processor
        state.current_model_name = model_name
        config.log.info("[multimodal] Ready: %s", model_name)

    except Exception as e:
        mcache.record_failed_load(model_name)
        config.log.exception("[multimodal] Failed to load %s", model_name)
        raise RuntimeError(f"Failed to load multimodal model '{model_name}': {e}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_multimodal_inference(model_name: str, messages, max_tokens: int, temperature: float) -> str:
    import torch as _torch

    cached = state.model_cache.get(model_name)
    if not cached:
        raise RuntimeError(f"Model '{model_name}' not loaded.")

    model = cached["model"]
    processor = state.mm_processor_cache.get(model_name) or cached["tokenizer"]
    inputs = build_mm_inputs(processor, messages)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=processor.tokenizer.eos_token_id
            if hasattr(processor, "tokenizer") else processor.eos_token_id,
    )
    with _torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tok.decode(output_ids[:, input_len:][0], skip_special_tokens=True).strip()


async def stream_multimodal_response(model_name: str, messages, max_tokens: int, temperature: float):
    import json
    from threading import Thread

    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        yield f"data: {json.dumps({'error': 'TextIteratorStreamer unavailable', 'done': True})}\n\n"
        return

    cached = state.model_cache.get(model_name)
    if not cached:
        yield f"data: {json.dumps({'error': f'Model {model_name!r} not loaded', 'done': True})}\n\n"
        return

    model = cached["model"]
    processor = state.mm_processor_cache.get(model_name) or cached["tokenizer"]
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    inputs = build_mm_inputs(processor, messages)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(
        target=model.generate,
        kwargs=dict(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0, streamer=streamer),
        daemon=True,
    )
    thread.start()

    try:
        for token_text in streamer:
            yield f"data: {json.dumps({'delta': {'content': token_text}, 'done': False})}\n\n"
        yield f"data: {json.dumps({'delta': {'content': ''}, 'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    finally:
        thread.join(timeout=5)
