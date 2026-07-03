# my_app/acquire/acquire_logic.py
"""Acquire service orchestration for direct-PDF URL retrieval.

The active production capability is HTTP probing and validated direct PDF download.
Non-direct-PDF URLs return a deterministic unsupported acquisition result.
"""

import logging
from typing import Optional

from .utils import (
    HTTP_PROBE_TIMEOUT,
    PDF_DOWNLOAD_TIMEOUT,
    PDF_RANGE_BYTES,
    _http_get_with_retry,
)
from .validation import _is_pdf_bytes

logger = logging.getLogger("AcquireService")


# ─────────────────────────────────────────────
# HTTP direct-PDF probe
# ─────────────────────────────────────────────

async def _http_probe(url: str) -> tuple[Optional[bytes], str]:
    """
    Attempt to retrieve a direct PDF via HTTP before classifying unsupported URL categories.
    This service does not perform browser-mediated acquisition.

    Strategy:
      1. Initial GET request — inspect Content-Type header
      2. GET Range bytes=0-1023 — confirm PDF magic bytes (%PDF-)
      3. Full GET if confirmed PDF

    Returns (pdf_bytes, method) or (None, "probe_miss").
    """
    import httpx
    try:
        head = await _http_get_with_retry(url, timeout=HTTP_PROBE_TIMEOUT)
        content_type = head.headers.get("content-type", "")

        if "pdf" in content_type.lower():
            logger.info(f"Probe: Content-Type signals PDF — fetching {url}")
            r = await _http_get_with_retry(url, timeout=PDF_DOWNLOAD_TIMEOUT)
            if r.status_code == 200 and _is_pdf_bytes(r.content):
                logger.info(f"Probe: direct PDF fetch success ({len(r.content) // 1024} KB)")
                return r.content, "probe_direct"

        range_resp = await _http_get_with_retry(
            url,
            headers={"Range": PDF_RANGE_BYTES},
            timeout=HTTP_PROBE_TIMEOUT
        )
        if range_resp.status_code in (200, 206):
            if _is_pdf_bytes(range_resp.content):
                logger.info(f"Probe: Range probe confirms PDF — fetching full {url}")
                full = await _http_get_with_retry(url, timeout=PDF_DOWNLOAD_TIMEOUT)
                if full.status_code == 200 and _is_pdf_bytes(full.content):
                    logger.info(f"Probe: full PDF fetch success ({len(full.content) // 1024} KB)")
                    return full.content, "probe_range"

    except httpx.TimeoutException:
        logger.info(f"Probe: timeout for {url} — direct PDF probe failed")
    except Exception as e:
        logger.info(f"Probe: {url} — {e} — direct PDF probe failed")

    return None, "probe_miss"


# ─────────────────────────────────────────────
# Category Detection
# ─────────────────────────────────────────────

def _detect_category(url: str) -> str:
    """
    Auto-detect acquisition category from URL patterns.
    Cat C is absorbed into Cat B (architecture rule).

    Returns: "A" | "B" | "E"
    Default: "A". Non-direct-PDF category handling returns unsupported.
    """
    url_lower = url.lower()

    social_domains = [
        "twitter.com", "x.com", "linkedin.com", "reddit.com",
        "substack.com", "medium.com", "facebook.com", "instagram.com",
        "threads.net", "bsky.app", "tumblr.com", "patreon.com"
    ]
    if any(d in url_lower for d in social_domains):
        return "E"

    locked_signals = [
        "doi.org", "pubmed", "ncbi.nlm.nih.gov", "jstor.org",
        "springer.com", "wiley.com", "elsevier.com", "sciencedirect.com",
        "nature.com", "science.org", "tandfonline.com", "sagepub.com",
        "researchgate.net", "acm.org", "ieee.org",
        ".gov/", ".gov.uk/", "congress.gov", "regulations.gov",
        "sec.gov", "federalregister.gov",
        "wsj.com", "ft.com", "nytimes.com", "thetimes.co.uk",
        "bloomberg.com", "economist.com", "newyorker.com"
    ]
    if any(s in url_lower for s in locked_signals):
        return "B"

    return "A"


# ─────────────────────────────────────────────
# Unsupported URL category handling
# ─────────────────────────────────────────────

async def _unsupported_non_direct_pdf(
        url: str,
        category: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    logger.info(
        "Unsupported URL acquisition category=%s url=%s trace_id=%s",
        category,
        url,
        trace_id,
    )
    return None, f"unsupported_non_direct_pdf_{category.lower()}"


# ─────────────────────────────────────────────
# Category Handlers
# ─────────────────────────────────────────────

async def _acquire_cat_a(
        url: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    """Cat A — direct PDF probe miss; browser acquisition unsupported."""
    return await _unsupported_non_direct_pdf(url, "A", trace_id)


async def _acquire_cat_b(
        url: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    """Cat B — access-controlled URL; unsupported by production acquisition."""
    return await _unsupported_non_direct_pdf(url, "B", trace_id)


async def _acquire_cat_e(
        url: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    """Cat E — social/login-walled URL; unsupported by production acquisition."""
    return await _unsupported_non_direct_pdf(url, "E", trace_id)