from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(tags=["voices"])

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _safe_resolve(root: str, rel: str) -> Path:
    root_p = Path(root).resolve()
    target = (root_p / rel).resolve()
    if root_p not in target.parents and target != root_p:
        raise HTTPException(status_code=400, detail="path escapes voice_presets_dir")
    return target


def _scan_presets(root: str) -> dict[str, Any]:
    root_p = Path(root)
    if not root_p.is_dir():
        return {"root": str(root_p), "languages": []}

    languages: list[dict[str, Any]] = []
    for lang_dir in sorted(p for p in root_p.iterdir() if p.is_dir()):
        voices: list[dict[str, Any]] = []
        for voice_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            files: list[dict[str, Any]] = []
            expressions: list[dict[str, Any]] = []
            for entry in sorted(voice_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in AUDIO_EXTS:
                    rel = entry.relative_to(root_p).as_posix()
                    files.append(
                        {
                            "name": entry.stem,
                            "filename": entry.name,
                            "format": entry.suffix.lower().lstrip("."),
                            "path": rel,
                        }
                    )
                elif entry.is_dir() and entry.name == "expressions":
                    for ex in sorted(entry.iterdir()):
                        if ex.is_file() and ex.suffix.lower() in AUDIO_EXTS:
                            rel = ex.relative_to(root_p).as_posix()
                            expressions.append(
                                {
                                    "name": ex.stem,
                                    "filename": ex.name,
                                    "format": ex.suffix.lower().lstrip("."),
                                    "path": rel,
                                }
                            )
            voices.append(
                {
                    "name": voice_dir.name,
                    "files": files,
                    "expressions": expressions,
                }
            )
        languages.append({"language": lang_dir.name, "voices": voices})

    return {"root": str(root_p.resolve()), "languages": languages}


@router.get("/voices", summary="List voice presets")
async def list_voices(request: Request) -> dict[str, Any]:
    cfg = request.app.state.cfg
    return _scan_presets(cfg.voice_presets_dir)


@router.get("/voices/file", summary="Serve a voice preset audio file")
async def get_voice_file(path: str, request: Request) -> FileResponse:
    cfg = request.app.state.cfg
    target = _safe_resolve(cfg.voice_presets_dir, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="preset file not found")
    ext = target.suffix.lower()
    media = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(ext, "application/octet-stream")
    return FileResponse(str(target), media_type=media, filename=target.name)


def load_preset_bytes(voice_presets_dir: str, rel_path: str) -> tuple[bytes, str]:
    target = _safe_resolve(voice_presets_dir, rel_path)
    if not target.is_file():
        raise HTTPException(status_code=400, detail=f"preset not found: {rel_path}")
    ext = target.suffix.lower().lstrip(".")
    if ext not in {e.lstrip(".") for e in AUDIO_EXTS}:
        raise HTTPException(status_code=400, detail=f"unsupported preset format: {ext}")
    return target.read_bytes(), ext
