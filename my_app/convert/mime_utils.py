# my_app/convert/mime_utils.py
"""
Convert Service — MIME detection and format routing constants.
"""

import logging
import magic

logger = logging.getLogger("ConvertService")

# ─────────────────────────────────────────────
# Format Routing Maps
# ─────────────────────────────────────────────

OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "application/msword",  # DOC
    "application/vnd.ms-powerpoint",  # PPT
    "application/vnd.ms-excel",  # XLS
    "application/vnd.oasis.opendocument.text",  # ODT
    "application/vnd.oasis.opendocument.presentation",  # ODP
    "application/vnd.oasis.opendocument.spreadsheet",  # ODS
    "application/rtf",
    "text/rtf",
}

EBOOK_MIMES = {
    "application/epub+zip",  # EPUB
    "application/x-mobipocket-ebook",  # MOBI
    "application/vnd.amazon.ebook",  # AZW
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


# ─────────────────────────────────────────────
# MIME Detection
# ─────────────────────────────────────────────

def _detect_mime(raw: bytes) -> str:
    """
    Detect MIME type from raw file bytes using libmagic.
    Authoritative over file extension — catches misnamed files.
    Falls back to 'application/octet-stream' on failure.
    """
    try:
        return magic.from_buffer(raw, mime=True)
    except Exception as e:
        logger.warning(f"MIME detection failed: {e}")
        return "application/octet-stream"