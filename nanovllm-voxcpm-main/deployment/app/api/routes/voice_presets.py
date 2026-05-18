from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["presets"])


def _find_presets_dir() -> Path | None:
    env = os.environ.get("NANOVLLM_VOICE_PRESETS_DIR", "")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    # Fallback: repo_root/voice_presets
    # __file__ = .../nanovllm-voxcpm-main/deployment/app/api/routes/voice_presets.py
    # parents[5] = .../nanovllm-voxcpm2-inference/ (or /app/ in Docker)
    candidate = Path(__file__).parents[5] / "voice_presets"
    return candidate if candidate.is_dir() else None


@router.get(
    "/voice_presets",
    summary="List available voice presets",
    responses={200: {"description": "Nested map of language -> voice_type -> emotion list"}},
)
async def list_voice_presets() -> Any:
    """Return all available voice preset audio files grouped by language and voice type.

    Each emotion entry contains `wav` and/or `mp3` keys pointing to static audio URLs
    that can be fetched directly from the browser for preview or base64-encoded for
    the /generate endpoint.
    """
    presets_dir = _find_presets_dir()
    if presets_dir is None:
        return {"presets": {}}

    result: dict[str, dict[str, list[dict[str, str]]]] = {}

    for lang_dir in sorted(presets_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name.startswith("."):
            continue
        lang = lang_dir.name
        result[lang] = {}

        for voice_dir in sorted(lang_dir.iterdir()):
            if not voice_dir.is_dir() or voice_dir.name.startswith("."):
                continue
            voice_type = voice_dir.name
            emotion_map: dict[str, dict[str, str]] = {}

            for f in sorted(voice_dir.iterdir()):
                if not f.is_file() or f.suffix not in (".wav", ".mp3"):
                    continue
                # Extract emotion from filename tail, e.g. "en_woman_happy.wav" -> "happy"
                parts = f.stem.rsplit("_", 1)
                emotion = parts[-1] if len(parts) >= 1 else f.stem
                rel = f.relative_to(presets_dir)
                url = f"/voice_presets/audio/{rel.as_posix()}"
                if emotion not in emotion_map:
                    emotion_map[emotion] = {}
                emotion_map[emotion][f.suffix.lstrip(".")] = url

            result[lang][voice_type] = [
                {"emotion": e, **formats} for e, formats in sorted(emotion_map.items())
            ]

    return {"presets": result}
