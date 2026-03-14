# my_app/acquire/validation.py
"""
Acquire Service — Post-acquisition PDF validation and response helpers.
"""

import logging
from fastapi import HTTPException
from fastapi.responses import Response

logger = logging.getLogger("AcquireService")


def _is_pdf_bytes(data: bytes) -> bool:
    """Check PDF magic bytes. Fast — no parsing."""
    return bool(data) and data[:5] == b"%PDF-"


def _pdf_response(pdf_bytes: bytes, url: str, method: str) -> Response:
    """Wrap PDF bytes in a standard Response with acquisition metadata headers."""
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "X-Acquire-Source": url[:500],
            "X-Acquire-Method": method,
        }
    )


class _SpanJsonResponse(Exception):
    """
    Dead code marker — OCR routing removed Phase 1b.
    Retained temporarily until the cleanup pass removes the unused
    exception and any unreachable branches referencing it.
    """

    def __init__(self, payload: dict):
        self.payload = payload


async def _post_acquisition_validate(
        pdf_bytes: bytes,
        url: str,
        method: str
) -> bytes:
    """
    Universal post-acquisition validation (all categories).

    Checks:
      1. Size > 0 and valid PDF magic bytes
      2. Page count > 0
      3. Text layer presence (informational only — no OCR routing)

    On empty/invalid → raises HTTPException 422.
    Returns validated pdf_bytes on pass.
    """
    if not pdf_bytes or not _is_pdf_bytes(pdf_bytes):
        raise HTTPException(status_code=422, detail=f"Invalid PDF produced from {url}")

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            doc.close()
            raise HTTPException(status_code=422, detail=f"Zero-page PDF from {url}")

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        page_count = doc.page_count
        doc.close()

        if not full_text.strip():
            logger.info(
                f"Validation: PDF valid but has no text layer from {url} "
                f"— returning PDF bytes for processor-side OCR"
            )
        else:
            logger.info(f"Validation: PDF ok — {page_count} pages, {len(full_text)} chars")

    except HTTPException:
        raise
    except ImportError:
        logger.warning("PyMuPDF not available — skipping deep validation")
    except Exception as e:
        logger.warning(f"Validation warning for {url}: {e}")

    return pdf_bytes