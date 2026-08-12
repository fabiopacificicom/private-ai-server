import os
import sys

# Add src/ to path so all modules import unchanged
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config  # noqa: F401 — env setup and backend imports run at import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from database import init_job_database
from loader import load_manifest
from routes.chat import router as chat_router
from routes.models import router as models_router
from routes.jobs import router as jobs_router
from routes.system import router as system_router
from routes.ollama import router as ollama_router

# Absolute path to the static UI folder (next to this file)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(
    title="Private AI Inference Server",
    version="1.0.0",
    description=(
        "GPU-optimized FastAPI server for serving Hugging Face LLMs with explicit "
        "download control, automatic quantization, and intelligent caching.\n\n"
        "**Backends:** vLLM (Linux), Transformers + bitsandbytes (Windows/Linux), "
        "llama-cpp-python (GGUF).\n\n"
        "**Primary endpoints:**\n"
        "- `POST /chat` — text chat (Ollama-compatible response)\n"
        "- `POST /chat/multimodal` — text + image/audio/video\n"
        "- `POST /pull` — download a model from HuggingFace\n"
        "- `GET /models` — list local + cached models\n"
        "- `GET /jobs/{id}` — poll background pull progress\n"
        "- `GET /health` / `/diag` / `/status` — server diagnostics"
    ),
    contact={"name": "Private AI Server", "url": "https://github.com/fabiopacificicom/private-ai-server"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "chat", "description": "Text chat inference (Ollama-compatible)."},
        {"name": "multimodal", "description": "Text + image/audio/video inference."},
        {"name": "models", "description": "List and pull models."},
        {"name": "jobs", "description": "Background pull job tracking."},
        {"name": "system", "description": "Health, diagnostics, and debug endpoints."},
    ],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log every unhandled exception with full traceback before returning 500."""
    config.log.exception(
        "Unhandled exception on %s %s", request.method, request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


init_job_database("jobs.db")
load_manifest()

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(jobs_router)
app.include_router(system_router)
app.include_router(ollama_router)


@app.get("/", include_in_schema=False)
async def index():
    """Serve the web chat UI at the root."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount static assets (CSS/JS) at /static
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Run: uvicorn app:app --host 0.0.0.0 --port 8005
