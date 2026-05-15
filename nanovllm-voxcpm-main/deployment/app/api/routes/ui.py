from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML_PATH = Path(__file__).parent.parent.parent / "static" / "demo.html"


@router.get("/", include_in_schema=False)
@router.get("/ui", summary="Interactive demo UI", response_class=HTMLResponse)
async def demo_ui() -> HTMLResponse:
    html = _HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
