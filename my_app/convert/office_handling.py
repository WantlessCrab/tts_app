# my_app/convert/office_handling.py
"""Convert service LibreOffice path and DOCX Mammoth/WeasyPrint fallback."""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import tempfile
from pathlib import Path

from fastapi import HTTPException

from .mime_utils import DOCX_MIMES
from .validation import _quality_gate

logger = logging.getLogger("ConvertService")

SOFFICE_PATH = None


def set_soffice_path(path: str) -> None:
    """Inject the configured LibreOffice executable path."""
    global SOFFICE_PATH
    SOFFICE_PATH = path


async def _convert_office(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str,
        trace_id: str = None,
) -> bytes:
    """
    Convert Office documents to canonical PDF.

    LibreOffice is the primary path. DOCX has a Mammoth/WeasyPrint fallback.
    Other Office formats return explicit unsupported-conversion errors if
    LibreOffice cannot produce a usable PDF.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        in_path.write_bytes(raw)

        logger.info("LibreOffice: converting %s [trace_id=%s]", filename, trace_id)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    SOFFICE_PATH,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(out_dir),
                    str(in_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            pdf_candidates = list(out_dir.glob("*.pdf"))

            if result.returncode == 0 and pdf_candidates:
                pdf_bytes = pdf_candidates[0].read_bytes()
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename, trace_id)
                if not issues:
                    logger.info("LibreOffice: quality gate passed for %s [trace_id=%s]", filename,
                                trace_id)
                    return pdf_bytes
                logger.warning("LibreOffice: quality gate failed — %s [trace_id=%s]", issues,
                               trace_id)
            else:
                logger.warning("LibreOffice: conversion failed — %s [trace_id=%s]",
                               result.stderr[:300], trace_id)

        except subprocess.TimeoutExpired:
            logger.error("LibreOffice: timeout on %s [trace_id=%s]", filename, trace_id)
        except Exception as exc:
            logger.error("LibreOffice: unexpected error — %s [trace_id=%s]", exc, trace_id)

    is_docx = ext == ".docx" or mime in DOCX_MIMES
    if is_docx:
        logger.info("Fallback: Mammoth to WeasyPrint for %s [trace_id=%s]", filename, trace_id)
        return await _mammoth_weasyprint(raw, filename, trace_id)

    raise HTTPException(
        status_code=422,
        detail=(
            f"LibreOffice did not produce a usable PDF from {filename}. "
            "No secondary converter is active for this office format."
        ),
    )


async def _mammoth_weasyprint(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """Convert DOCX to semantic HTML with Mammoth, then render PDF with WeasyPrint."""
    try:
        try:
            import mammoth
            from weasyprint import HTML as WeasyHTML
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"DOCX fallback dependencies unavailable: {exc}",
            ) from exc

        with io.BytesIO(raw) as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
            if result.messages:
                for msg in result.messages:
                    logger.info("Mammoth [%s]: %s [trace_id=%s]", filename, msg.message, trace_id)

        pdf_bytes = await asyncio.to_thread(WeasyHTML(string=html).write_pdf)
        logger.info("Mammoth to WeasyPrint: %s -> %s KB [trace_id=%s]", filename,
                    len(pdf_bytes) // 1024, trace_id)
        return pdf_bytes

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Mammoth to WeasyPrint failed for %s: %s [trace_id=%s]", filename, exc,
                     trace_id)
        raise HTTPException(
            status_code=500,
            detail=f"All DOCX conversion paths failed for {filename}: {exc}",
        ) from exc