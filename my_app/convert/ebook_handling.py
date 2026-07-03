# my_app/convert/ebook_handling.py
"""Convert service Calibre path and EPUB/WeasyPrint fallback."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from fastapi import HTTPException

from .validation import _quality_gate

logger = logging.getLogger("ConvertService")


async def _convert_ebook(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str,
        trace_id: str = None,
) -> bytes:
    """
    Route EPUB/MOBI/AZW through Calibre first.

    Fallback is intentionally limited to EPUB HTML extraction. If Calibre cannot
    produce a usable PDF for MOBI/AZW/AZW3, the service returns an explicit
    unsupported-conversion error instead of advertising an inactive layout path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_path = Path(tmpdir) / "output.pdf"
        in_path.write_bytes(raw)

        logger.info("Calibre: converting %s [trace_id=%s]", filename, trace_id)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ebook-convert", str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode == 0 and out_path.exists():
                pdf_bytes = out_path.read_bytes()
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename, trace_id)
                if not issues:
                    logger.info("Calibre: quality gate passed for %s [trace_id=%s]", filename,
                                trace_id)
                    return pdf_bytes
                logger.warning("Calibre: quality gate failed — %s [trace_id=%s]", issues, trace_id)
            else:
                logger.warning("Calibre: conversion failed — %s [trace_id=%s]", result.stderr[:300],
                               trace_id)

        except subprocess.TimeoutExpired:
            logger.error("Calibre: timeout on %s [trace_id=%s]", filename, trace_id)
        except Exception as exc:
            logger.error("Calibre: unexpected error — %s [trace_id=%s]", exc, trace_id)

    is_epub = ext == ".epub" or mime == "application/epub+zip"
    if is_epub:
        logger.info("Fallback: EPUB HTML extraction to WeasyPrint for %s [trace_id=%s]", filename,
                    trace_id)
        return await _epub_html_weasyprint(raw, filename, trace_id)

    raise HTTPException(
        status_code=422,
        detail=(
            f"Calibre did not produce a usable PDF from {filename}. "
            "No secondary converter is active for this ebook format."
        ),
    )


async def _epub_html_weasyprint(raw: bytes, filename: str, trace_id: str = None) -> bytes:
    """Convert EPUB HTML spine content to PDF through WeasyPrint."""
    try:
        try:
            import ebooklib
            from ebooklib import epub
            from weasyprint import HTML as WeasyHTML
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"EPUB fallback dependencies unavailable: {exc}",
            ) from exc

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
                "EPUB to WeasyPrint: %s -> %s KB [trace_id=%s]",
                filename,
                len(pdf_bytes) // 1024,
                trace_id,
            )
            return pdf_bytes

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("EPUB to WeasyPrint failed for %s: %s [trace_id=%s]", filename, exc, trace_id)
        raise HTTPException(
            status_code=500,
            detail=f"All EPUB conversion paths failed for {filename}: {exc}",
        ) from exc