# ~/TTS/my_app/gateway_router.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY ROUTER — Routing decisions and orchestration sequence
# Generates trace_id. Classifies input. Calls correct client. Writes PDF.
# Triggers processor. Returns GatewayResult.
# No HTTP handling here — that lives in gateway_clients.py.
# ════════════════════════════════════════════════════════════════════════════════

import uuid
import logging
import aiofiles
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from gateway_contracts import GatewayResult
from gateway_clients import call_acquire, call_convert, call_processor

logger = logging.getLogger("GatewayRouter")

INPUT_DIR = Path("/workspace/pdf_input")


async def route_ingest(
        url: Optional[str] = None,
        file: Optional[UploadFile] = None,
        supplied_trace_id: Optional[str] = None,
) -> GatewayResult:
    """
    Single orchestration entry point for all ingest paths.

    Path A: URL provided           → call_acquire → write PDF → call_processor
    Path B: Office/Ebook file      → call_convert → write PDF → call_processor
    Path C: PDF file uploaded      → write PDF directly → call_processor

    Raises ValueError on invalid input or non-PDF response from any service.
    """
    trace_id = supplied_trace_id or str(uuid.uuid4())
    pdf_filename = f"{trace_id}.pdf"
    pdf_path = INPUT_DIR / pdf_filename
    source_filename: Optional[str] = None

    # ── Path A: URL ───────────────────────────────────────────────────────────
    if url:
        source_filename = url
        logger.info(f"[{trace_id}] Route: URL → acquire")
        pdf_bytes = await call_acquire(url, trace_id)

    # ── Path B / C: File upload ───────────────────────────────────────────────
    elif file:
        source_filename = file.filename or "upload"
        file_bytes = await file.read()
        content_type = (file.content_type or "").lower()
        ext = Path(source_filename).suffix.lower()

        if content_type == "application/pdf" or ext == ".pdf":
            logger.info(f"[{trace_id}] Route: direct PDF upload → write")
            pdf_bytes = file_bytes
        else:
            logger.info(f"[{trace_id}] Route: {content_type} → convert")
            pdf_bytes = await call_convert(file_bytes, source_filename, trace_id)

    else:
        raise ValueError("Must provide either url or file")

    # ── Write canonical PDF to disk ───────────────────────────────────────────
    logger.info(f"[{trace_id}] Writing {len(pdf_bytes) // 1024} KB → {pdf_path}")
    async with aiofiles.open(pdf_path, "wb") as f:
        await f.write(pdf_bytes)

    # ── Trigger processor ─────────────────────────────────────────────────────
    result = await call_processor(pdf_filename, trace_id)

    return GatewayResult(
        book_id=result.get("book_id", trace_id),
        trace_id=trace_id,
        status=result.get("status", "processing_started"),
        source_filename=source_filename,
    )