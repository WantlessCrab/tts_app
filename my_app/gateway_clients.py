# ~/TTS/my_app/gateway_clients.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY CLIENTS — Outbound HTTP calls to downstream services
# Every function injects X-Trace-ID. Every function validates content-type.
# No routing logic here. No fallback logic here.
# ════════════════════════════════════════════════════════════════════════════════

import os
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("GatewayClients")

ACQUIRE_SERVICE_URL = os.getenv("ACQUIRE_SERVICE_URL", "http://host.docker.internal:8005")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")


@dataclass(frozen=True)
class PDFServiceResult:
    """PDF bytes plus downstream provenance headers captured by the gateway."""

    content: bytes
    content_type: str
    method: str
    source: Optional[str] = None
    headers: dict[str, str] | None = None


def _metadata_headers(response: httpx.Response) -> dict[str, str]:
    prefixes = ("x-acquire-", "x-convert-", "x-source-", "x-trace-")
    return {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower().startswith(prefixes)
    }


async def call_acquire(url: str, trace_id: str) -> PDFServiceResult:
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
    headers = _metadata_headers(r)
    return PDFServiceResult(
        content=r.content,
        content_type=content_type,
        method=headers.get("x-acquire-method") or "acquire_service",
        source=headers.get("x-acquire-source") or url,
        headers=headers,
    )


async def call_convert(file_bytes: bytes, filename: str, trace_id: str) -> PDFServiceResult:
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
    headers = _metadata_headers(r)
    return PDFServiceResult(
        content=r.content,
        content_type=content_type,
        method=headers.get("x-convert-method") or "convert_service",
        source=headers.get("x-source-file") or filename,
        headers=headers,
    )


async def call_processor(
        pdf_filename: str,
        trace_id: str,
        *,
        document_id: str | None = None,
        job_id: str | None = None,
        audiobook_id: str | None = None,
) -> dict:
    """
    POST to pdf-processor to trigger full pipeline for a named file.
    Returns processor response dict.

    X-Document-ID, X-Job-ID, and X-Audiobook-ID are provided by the
    canonical gateway ingest path and passed through as processor lineage.
    """
    logger.info(f"[{trace_id}] call_processor: {pdf_filename}")
    headers = {"X-Trace-ID": trace_id}
    if document_id:
        headers["X-Document-ID"] = document_id
    if job_id:
        headers["X-Job-ID"] = job_id
    if audiobook_id:
        headers["X-Audiobook-ID"] = audiobook_id

    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.post(
            f"{PDF_SERVICE_URL}/api/v1/process/{pdf_filename}",
            headers=headers,
        )
        r.raise_for_status()

    return r.json()