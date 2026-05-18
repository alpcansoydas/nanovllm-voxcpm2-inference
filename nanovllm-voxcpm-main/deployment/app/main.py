from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.api import api_router
from app.core.config import load_config
from app.core.lifespan import build_lifespan
from app.core.metrics import install_metrics

_STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def _find_voice_presets_dir() -> Path | None:
    env = os.environ.get("NANOVLLM_VOICE_PRESETS_DIR", "")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    # __file__ = .../nanovllm-voxcpm-main/deployment/app/main.py
    # parents[3] = .../nanovllm-voxcpm2-inference/ (or /app/ in Docker)
    candidate = Path(__file__).parents[3] / "voice_presets"
    return candidate if candidate.is_dir() else None


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="nano-vllm VoxCPM Service",
        version="0.1.0",
        description=(
            "Production-oriented FastAPI wrapper for nano-vllm-voxcpm. "
            "See /docs for interactive API docs and /openapi.json for the OpenAPI schema. "
            "Demo UI is available at /ui."
        ),
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "info", "description": "Model and instance metadata."},
            {"name": "metrics", "description": "Prometheus metrics."},
            {"name": "lora", "description": "Runtime LoRA adapter management."},
            {"name": "latents", "description": "Encode prompt audio to prompt latents."},
            {"name": "generation", "description": "Text-to-speech generation (streaming MP3)."},
            {"name": "presets", "description": "Voice preset audio files."},
        ],
        lifespan=build_lifespan(cfg),
    )
    app.state.cfg = cfg
    install_metrics(app)
    app.include_router(api_router)

    # Mount voice preset audio files as static assets
    voice_presets_dir = _find_voice_presets_dir()
    if voice_presets_dir is not None:
        app.mount(
            "/voice_presets/audio",
            StaticFiles(directory=str(voice_presets_dir)),
            name="voice_presets_audio",
        )
        logger.info("Voice presets mounted from %s", voice_presets_dir)
    else:
        logger.warning("Voice presets directory not found; /voice_presets/audio will not be available")

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def demo_ui() -> HTMLResponse:
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    return app


app = create_app()
