# my_app/convert/mime_utils.py
"""Convert service MIME detection and format routing constants."""

from __future__ import annotations

import logging

logger = logging.getLogger("ConvertService")

OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/rtf",
    "text/rtf",
}

EBOOK_MIMES = {
    "application/epub+zip",
    "application/x-mobipocket-ebook",
    "application/vnd.amazon.ebook",
}

OFFICE_EXTENSIONS = {
    ".docx", ".doc", ".odt", ".pptx", ".ppt",
    ".odp", ".xlsx", ".xls", ".ods", ".rtf"
}

EBOOK_EXTENSIONS = {".epub", ".mobi", ".azw", ".azw3"}

DOCX_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}


def _detect_mime(raw: bytes) -> str:
    """Detect MIME type from bytes, degrading safely when libmagic is unavailable."""
    try:
        import magic
    except Exception as exc:
        logger.warning("python-magic unavailable; MIME falls back to application/octet-stream: %s",
                       exc)
        return "application/octet-stream"

    try:
        return magic.from_buffer(raw, mime=True)
    except Exception as exc:
        logger.warning("MIME detection failed: %s", exc)
        return "application/octet-stream"