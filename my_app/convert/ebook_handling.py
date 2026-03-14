# my_app/convert/ebook_handling.py
"""
Convert Service — Calibre conversion path, EPUB/WeasyPrint fallback, Docling stub.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import ebooklib
from ebooklib import epub
from weasyprint import HTML as WeasyHTML
from fastapi import HTTPException

from .validation import _quality_gate

logger = logging.getLogger("ConvertService")


async def _convert_ebook(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str,
        trace_id: str = None
) -> bytes:
    """
    Route B: EPUB / MOBI / AZW

    Step 1 — Calibre → canonical PDF
    Step 2 — Quality gate
    Step 3 — On fail:
               EPUB → HTML extraction → WeasyPrint
               MOBI/AZW → Docling fallback stub

    subprocess.run wrapped in asyncio.to_thread — non-blocking.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_path = Path(tmpdir) / "output.pdf"
        in_path.write_bytes(raw)

        logger.info(f"Calibre: converting {filename} [trace_id={trace_id}]")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ebook-convert", str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=180
            )

            if result.returncode == 0 and out_path.exists():
                pdf_bytes = out_path.read_bytes()
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename, trace_id)
                if not issues:
                    logger.info(
                        f"Calibre: quality gate passed for {filename} [trace_id={trace_id}]")
                    return pdf_bytes
                else:
                    logger.warning(f"Calibre: quality gate failed — {issues} [trace_id={trace_id}]")
            else:
                logger.warning(
                    f"Calibre: conversion failed — {result.stderr[:300]} [trace_id={trace_id}]")

        except subprocess.TimeoutExpired:
            logger.error(f"Calibre: timeout on {filename} [trace_id={trace_id}]")
        except Exception as e:
            logger.error(f"Calibre: unexpected error — {e} [trace_id={trace_id}]")

    # Fallback routing
    is_epub = (ext == ".epub" or mime == "application/epub+zip")
    if is_epub:
        logger.info(
            f"Fallback: EPUB HTML extraction → WeasyPrint for {filename} [trace_id={trace_id}]")
        return await _epub_html_weasyprint(raw, filename, trace_id)
    else:
        logger.info(f"Fallback: Docling stub for {filename} (MOBI/AZW) [trace_id={trace_id}]")
        return await _docling_fallback(raw, filename, trace_id)


async def _epub_html_weasyprint(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """
    EPUB → extract HTML content via ebooklib → WeasyPrint → PDF.
    Used when Calibre PDF conversion produces unacceptable output.
    Preserves spine order from EPUB structure.
    WeasyPrint render is CPU-bound — run in thread pool.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            book = epub.read_epub(tmp_path)
            html_parts = []

            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content().decode("utf-8", errors="replace")
                html_parts.append(content)

            if not html_parts:
                raise ValueError("No HTML content items found in EPUB")

            combined = "\n".join(html_parts)
            pdf_bytes = await asyncio.to_thread(WeasyHTML(string=combined).write_pdf)
            logger.info(
                f"EPUB→WeasyPrint: {filename} → {len(pdf_bytes) // 1024} KB [trace_id={trace_id}]")
            return pdf_bytes

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"EPUB→WeasyPrint failed for {filename}: {e} [trace_id={trace_id}]")
        raise HTTPException(
            status_code=500,
            detail=f"All EPUB conversion paths failed for {filename}: {e}"
        )


async def _docling_fallback(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """
    Last-resort fallback — Docling native ingestion → span JSON.
    Docling runs in its own container (port 8007, future build).
    Raises 422 until that container is live.

    When layout container is implemented:
      POST raw bytes → layout service (port 8007)
      → receive span JSON → normalize_to_raw_spans() → extraction pipeline
    """
    logger.error(
        f"All PDF conversion paths exhausted for {filename}. "
        f"Docling fallback not yet implemented (layout container pending). "
        f"Manual review required. [trace_id={trace_id}]"
    )
    raise HTTPException(
        status_code=422,
        detail=(
            f"Could not produce a usable PDF from {filename}. "
            f"All conversion paths exhausted. "
            f"Docling layout fallback is pending (layout container not yet built)."
        )
    )