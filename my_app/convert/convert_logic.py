# my_app/convert/convert_logic.py
"""
Convert Service — Top-level MIME routing orchestration.
"""

import logging
from pathlib import Path

from fastapi import HTTPException

from .mime_utils import (
    OFFICE_MIMES, OFFICE_EXTENSIONS,
    EBOOK_MIMES, EBOOK_EXTENSIONS,
    _detect_mime,
)
from .office_handling import _convert_office
from .ebook_handling import _convert_ebook

logger = logging.getLogger("ConvertService")


async def route_conversion(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """
    Detect format via MIME + extension and route to the correct conversion path.
    Returns canonical PDF bytes.
    """
    ext = Path(filename).suffix.lower()
    mime = _detect_mime(raw)

    logger.info(f"MIME: {mime} | ext: {ext} [trace_id={trace_id}]")

    if mime in OFFICE_MIMES or ext in OFFICE_EXTENSIONS:
        return await _convert_office(raw, filename, ext, mime, trace_id)
    elif mime in EBOOK_MIMES or ext in EBOOK_EXTENSIONS:
        return await _convert_ebook(raw, filename, ext, mime, trace_id)
    else:
        logger.warning(f"Unsupported format: mime={mime} ext={ext} [trace_id={trace_id}]")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: mime={mime}, ext={ext}"
        )