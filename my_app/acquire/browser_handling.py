# my_app/acquire/browser_handling.py
"""Explicit unsupported URL-acquisition boundary.

The acquire service production path is direct-PDF retrieval only. This module
returns structured unsupported results for any code path that still asks for
browser-mediated acquisition.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("AcquireService")


async def acquire_with_browser(
        url: str,
        context_type: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    """Return a stable unsupported result for non-direct-PDF URL acquisition."""
    logger.info(
        "Browser-mediated acquisition unsupported for url=%s context=%s trace_id=%s",
        url,
        context_type,
        trace_id,
    )
    return None, "browser_acquisition_unsupported"