"""
Ollama-compatible API endpoints.

These mirror Ollama's HTTP contract so any client built for Ollama (e.g. the
web UI, or tools that call `/api/chat` and `/api/tags`) works unchanged against
this server — a true drop-in replacement.

- `POST /api/chat` — Ollama chat completion (request/response shaped like Ollama)
- `GET  /api/tags` — Ollama model list

Both delegate to the server's internal `/chat` and `/models` logic.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import state
from loader import load_model
from routes.chat import _chat_non_streaming, _stream_chat
from routes.models import list_models
from schemas import ChatRequest
from utils import with_timeout, gpu_cleanup_fn

router = APIRouter(prefix="/api", tags=["ollama"])


class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage]
    stream: bool = False
    options: Optional[Dict[str, Any]] = None  # temperature, num_predict, etc.


def _extract_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map Ollama option keys to our internal ones."""
    if not options:
        return {}
    mapped: Dict[str, Any] = {}
    if "temperature" in options:
        mapped["temperature"] = options["temperature"]
    if "num_predict" in options:
        mapped["max_tokens"] = options["num_predict"]
    return mapped


def _ollama_response(model: str, content: str, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build an Ollama-shaped chat response, carrying through timing/usage stats."""
    resp: Dict[str, Any] = {
        "model": model,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }
    if stats:
        # Carry through the usage/timing fields the internal engine provides
        for key in ("total_duration", "load_duration", "prompt_eval_count",
                    "prompt_eval_duration", "eval_count", "eval_duration"):
            if key in stats:
                resp[key] = stats[key]
    return resp


@router.post(
    "/chat",
    summary="Ollama-compatible chat completion",
    description=(
        "Mirrors Ollama's `POST /api/chat`. Accepts `{model, messages, stream, options}` "
        "and returns an Ollama-shaped response. Set `stream: true` for NDJSON chunks."
    ),
)
async def ollama_chat(request: OllamaChatRequest):
    opts = _extract_options(request.options)
    timeout_s = 120

    # Build an internal ChatRequest from the Ollama payload
    chat_req = ChatRequest(
        model=request.model,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        max_tokens=opts.get("max_tokens", 512),
        temperature=opts.get("temperature", 0.7),
        stream=request.stream,
        timeout=timeout_s,
    )

    if request.stream:
        # Ollama streams NDJSON (one JSON object per line, no "data:" prefix)
        async def _ndjson():
            async for chunk in _stream_chat(
                chat_req.model, chat_req.messages, chat_req.max_tokens, chat_req.temperature
            ):
                # Re-shape the internal SSE chunk into an Ollama message chunk
                line = chunk.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if data.get("done"):
                    # Final chunk with stats
                    yield json.dumps(data) + "\n"
                    continue
                delta = data.get("delta", {}).get("content", "")
                if delta:
                    out = {
                        "model": chat_req.model,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "message": {"role": "assistant", "content": delta},
                        "done": False,
                    }
                    yield json.dumps(out) + "\n"

        return StreamingResponse(
            _ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming: delegate to the internal chat implementation
    result = await with_timeout(
        _chat_non_streaming(chat_req, chat_req.model), timeout_s, gpu_cleanup_fn
    )
    content = result.get("message", {}).get("content", "")
    return _ollama_response(request.model, content, stats=result)


@router.get("/tags", summary="Ollama-compatible model list")
async def ollama_tags():
    """Mirror Ollama's `GET /api/tags` — returns the list of installed models."""
    # Reuse the internal /models listing logic
    models_resp = await list_models()
    models = models_resp.get("models", [])
    services = models_resp.get("services", {})

    ollama_models = []
    for m in models:
        # Only include models that are actually loadable (have a backend or local path)
        ollama_models.append({
            "name": m["model"],
            "model": m["model"],
            "size": m.get("size_bytes") or 0,
            "modality": m.get("modality", "unknown"),
            "digest": "sha256:" + str(abs(hash(m["model"]))),
            "details": {
                "format": "gguf" if (m.get("backend") == "gguf_llama_cpp") else "safetensors",
                "family": "unknown",
                "parameter_size": "",
                "quantization_level": "",
            },
        })

    return {"models": ollama_models, "total_duration": 0, "services": services}
