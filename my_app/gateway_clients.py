# ~/TTS/my_app/gateway_clients.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY CLIENTS — Outbound HTTP calls to downstream services
# Every function injects X-Trace-ID. Every function validates content-type.
# No routing logic here. No fallback logic here.
# ════════════════════════════════════════════════════════════════════════════════

import os
import logging
import httpx

logger = logging.getLogger("GatewayClients")

ACQUIRE_SERVICE_URL = os.getenv("ACQUIRE_SERVICE_URL", "http://host.docker.internal:8005")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")


async def call_acquire(url: str, trace_id: str) -> bytes:
    """
    POST to acquire service. Returns PDF bytes.
    Raises ValueError if response is not application/pdf.
    """
    logger.info(f"[{trace_id}] call_acquire: {url}")
    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as c:
        r = await c.post(
            f"{ACQUIRE_SERVICE_URL}/acquire",
            json={"url": url, "category": "auto"},
            headers={"X-Trace-ID": trace_id},
        )
        r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    if "application/pdf" not in content_type:
        raise ValueError(
            f"[{trace_id}] Acquire returned non-PDF content-type: {content_type!r}"
        )

    logger.info(f"[{trace_id}] call_acquire: received {len(r.content) // 1024} KB PDF")
    return r.content


async def call_convert(file_bytes: bytes, filename: str, trace_id: str) -> bytes:
    """
    POST to convert service as multipart upload. Returns PDF bytes.
    Raises ValueError if response is not application/pdf.
    """
    logger.info(f"[{trace_id}] call_convert: {filename}")
    async with httpx.AsyncClient(timeout=300.0) as c:
        r = await c.post(
            f"{CONVERT_SERVICE_URL}/convert",
            files={"file": (filename, file_bytes)},
            headers={"X-Trace-ID": trace_id},
        )
        r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    if "application/pdf" not in content_type:
        raise ValueError(
            f"[{trace_id}] Convert returned non-PDF content-type: {content_type!r}"
        )

    logger.info(f"[{trace_id}] call_convert: received {len(r.content) // 1024} KB PDF")
    return r.content


async def call_processor(pdf_filename: str, trace_id: str) -> dict:
    """
    POST to pdf-processor to trigger full pipeline for a named file.
    Returns processor response dict.
    """
    logger.info(f"[{trace_id}] call_processor: {pdf_filename}")
    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.post(
            f"{PDF_SERVICE_URL}/api/v1/process/{pdf_filename}",
            headers={"X-Trace-ID": trace_id},
        )
        r.raise_for_status()

    return r.json()