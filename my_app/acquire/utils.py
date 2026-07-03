# my_app/acquire/utils.py
"""Acquire service constants, HTTP client, retry helper, and failure response."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Optional

import httpx
from fastapi.responses import JSONResponse

logger = logging.getLogger("AcquireService")

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")

HTTP_PROBE_TIMEOUT = 15
PDF_DOWNLOAD_TIMEOUT = 60
PDF_RANGE_BYTES = "bytes=0-1023"

HTTP_RETRY_MAX = 3
HTTP_RETRY_DELAY = 1.0

client = httpx.AsyncClient(
    timeout=HTTP_PROBE_TIMEOUT,
    follow_redirects=True,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    },
)


async def _http_get_with_retry(url: str, **kwargs) -> httpx.Response:
    """HTTP GET with bounded retry on transient connect/read failures."""
    import asyncio

    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_RETRY_MAX):
        try:
            return await client.get(url, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout) as exc:
            last_exc = exc
            if attempt < HTTP_RETRY_MAX - 1:
                wait = HTTP_RETRY_DELAY * (attempt + 1)
                logger.info(
                    "HTTP retry %s/%s: %s — %s — waiting %.1fs",
                    attempt + 1,
                    HTTP_RETRY_MAX,
                    url,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _dead_letter(url: str, category: str, reason: str,
                 trace_id: Optional[str] = None) -> JSONResponse:
    """Return a structured acquisition failure record for caller-side handling."""
    record = {
        "status": "failed",
        "url": url,
        "category": category,
        "reason": reason,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.error("Dead letter: %s", record)
    return JSONResponse(status_code=422, content=record)