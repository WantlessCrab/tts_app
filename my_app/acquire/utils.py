# my_app/acquire/utils.py
"""
Acquire Service — Shared constants, HTTP client, and stateless utility functions.
All other acquire modules import constants and client from here.
"""

import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi.responses import JSONResponse

logger = logging.getLogger("AcquireService")

# ─────────────────────────────────────────────
# Environment Config
# ─────────────────────────────────────────────

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")

# ─────────────────────────────────────────────
# Standardized Container Paths
# ─────────────────────────────────────────────

STATIC_DIR = Path("/app/static")
WORKSPACE_DIR = Path("/workspace")
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
INPUT_DIR = WORKSPACE_DIR / "pdf_input"
PDF_CACHE_DIR = WORKSPACE_DIR / "pdf_cache"

READABILITY_JS = STATIC_DIR / "Readability.js"
SESSION_DIR = Path("/app/sessions")

# ─────────────────────────────────────────────
# Acquisition Timeouts
# ─────────────────────────────────────────────

HTTP_PROBE_TIMEOUT = 15  # seconds — HEAD/range probe
PAGE_LOAD_TIMEOUT = 30000  # ms — Playwright networkidle
PDF_DOWNLOAD_TIMEOUT = 60  # seconds — PDF stream download
STEALTH_RETRY_DELAY = 2.5  # seconds — delay before stealth retry

# ─────────────────────────────────────────────
# Retry Config
# ─────────────────────────────────────────────

MAX_CAPTCHA_RETRIES = 2  # max stealth retry loops inside unlock sequence
HTTP_RETRY_MAX = 3
HTTP_RETRY_DELAY = 1.0  # seconds base delay (multiplied by attempt number)

# ─────────────────────────────────────────────
# Probe / Validation Thresholds
# ─────────────────────────────────────────────

PDF_RANGE_BYTES = "bytes=0-1023"
MIN_PDF_BYTES = 1024

# ─────────────────────────────────────────────
# PDF Link Heuristic — preferred content signal terms
# ─────────────────────────────────────────────

PDF_LINK_PREFERRED_TERMS = {
    "bill", "report", "document", "paper", "publication",
    "filing", "release", "statement", "brief", "memo"
}

# ─────────────────────────────────────────────
# Browser UA
# ─────────────────────────────────────────────

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ─────────────────────────────────────────────
# Shared Async HTTP Client
# ─────────────────────────────────────────────

client = httpx.AsyncClient(
    timeout=HTTP_PROBE_TIMEOUT,
    follow_redirects=True,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    headers={"User-Agent": CHROME_UA}
)


# ─────────────────────────────────────────────
# HTTP Retry Wrapper
# ─────────────────────────────────────────────

async def _http_get_with_retry(url: str, **kwargs) -> httpx.Response:
    """
    HTTP GET with exponential backoff retry on transient network failures.
    Used for all network retrieval calls in the acquire service.
    Retries on ConnectError and ReadTimeout only — not on HTTP status errors.

    Max retries: HTTP_RETRY_MAX (default 3)
    Delay: HTTP_RETRY_DELAY * attempt (1s, 2s, 3s)
    """
    import asyncio
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_RETRY_MAX):
        try:
            return await client.get(url, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exc = e
            if attempt < HTTP_RETRY_MAX - 1:
                wait = HTTP_RETRY_DELAY * (attempt + 1)
                logger.info(
                    f"HTTP retry {attempt + 1}/{HTTP_RETRY_MAX}: {url} — {e} — waiting {wait}s"
                )
                await asyncio.sleep(wait)
    raise last_exc


# ─────────────────────────────────────────────
# URL Utilities
# ─────────────────────────────────────────────

def _extract_doi(url: str) -> Optional[str]:
    """Extract DOI from URL if present. Returns DOI string or None."""
    match = re.search(r'10\.\d{4,}/[^\s&?#]+', url)
    return match.group(0) if match else None


def _extract_domain(url: str) -> str:
    """Extract base domain from URL for session keying."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = parsed.netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else parsed.netloc
    except Exception:
        return "unknown"


def _get_session_path(domain: str) -> Path:
    """Return path to stored Playwright storage_state JSON for a domain."""
    safe_domain = re.sub(r'[^\w\-\.]', '_', domain)
    return SESSION_DIR / f"{safe_domain}.json"


# ─────────────────────────────────────────────
# Dead-Letter Response Helper
# ─────────────────────────────────────────────

def _dead_letter(url: str, category: str, reason: str,
                 trace_id: Optional[str] = None) -> JSONResponse:
    """
    Return a structured failure record when all acquisition paths fail.
    Dead-letter records are logged and returned as 422 JSON for the caller
    to route to manual review or the dead-letter log.
    """
    record = {
        "status": "failed",
        "url": url,
        "category": category,
        "reason": reason,
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    logger.error(f"Dead letter: {record}")
    return JSONResponse(status_code=422, content=record)