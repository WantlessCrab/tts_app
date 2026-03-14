# my_app/convert/office_handling.py
"""
Convert Service — LibreOffice conversion path and DOCX Mammoth/WeasyPrint fallback.
"""

import io
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import mammoth
from weasyprint import HTML as WeasyHTML
from fastapi import HTTPException

from .mime_utils import DOCX_MIMES
from .validation import _quality_gate

logger = logging.getLogger("ConvertService")

SOFFICE_PATH = None  # injected from api.py at startup via set_soffice_path()


def set_soffice_path(path: str) -> None:
    """Called from api.py startup to inject the configured soffice path."""
    global SOFFICE_PATH
    SOFFICE_PATH = path


async def _convert_office(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str,
        trace_id: str = None
) -> bytes:
    """
    Route A: DOCX / ODT / PPTX / XLSX / RTF / ODS

    Step 1 — LibreOffice headless → canonical PDF
    Step 2 — Quality gate
    Step 3 — On fail:
               DOCX → Mammoth → WeasyPrint
               Other → Docling fallback stub

    subprocess.run wrapped in asyncio.to_thread — non-blocking.
    """
    from .ebook_handling import _docling_fallback  # avoid circular at module level

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        in_path.write_bytes(raw)

        logger.info(f"LibreOffice: converting {filename} [trace_id={trace_id}]")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    SOFFICE_PATH,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", str(out_dir),
                    str(in_path)
                ],
                capture_output=True, text=True, timeout=120
            )
            pdf_candidates = list(out_dir.glob("*.pdf"))

            if result.returncode == 0 and pdf_candidates:
                pdf_bytes = pdf_candidates[0].read_bytes()
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename, trace_id)
                if not issues:
                    logger.info(
                        f"LibreOffice: quality gate passed for {filename} [trace_id={trace_id}]")
                    return pdf_bytes
                else:
                    logger.warning(
                        f"LibreOffice: quality gate failed — {issues} [trace_id={trace_id}]")
            else:
                logger.warning(
                    f"LibreOffice: conversion failed — {result.stderr[:300]} [trace_id={trace_id}]")

        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice: timeout on {filename} [trace_id={trace_id}]")
        except Exception as e:
            logger.error(f"LibreOffice: unexpected error — {e} [trace_id={trace_id}]")

    # Fallback routing
    is_docx = (ext == ".docx" or mime in DOCX_MIMES)
    if is_docx:
        logger.info(f"Fallback: Mammoth → WeasyPrint for {filename} [trace_id={trace_id}]")
        return await _mammoth_weasyprint(raw, filename, trace_id)
    else:
        logger.info(
            f"Fallback: Docling stub for {filename} (non-DOCX office format) [trace_id={trace_id}]")
        return await _docling_fallback(raw, filename, trace_id)


async def _mammoth_weasyprint(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """
    DOCX → Mammoth semantic HTML → WeasyPrint → PDF.
    Used when LibreOffice reflow produces unacceptable output.
    WeasyPrint render is CPU-bound — run in thread pool.
    """
    try:
        with io.BytesIO(raw) as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
            if result.messages:
                for msg in result.messages:
                    logger.info(f"Mammoth [{filename}]: {msg.message} [trace_id={trace_id}]")

        pdf_bytes = await asyncio.to_thread(WeasyHTML(string=html).write_pdf)
        logger.info(
            f"Mammoth→WeasyPrint: {filename} → {len(pdf_bytes) // 1024} KB [trace_id={trace_id}]")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Mammoth→WeasyPrint failed for {filename}: {e} [trace_id={trace_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"All DOCX conversion paths failed for {filename}: {e}"
        )