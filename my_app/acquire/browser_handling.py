# my_app/acquire/browser_handling.py
"""
Acquire Service — Playwright browser management, CDP intercept, DOM stabilization.

Public contract:
  acquire_with_browser()  — sole public entry point, signature frozen
                            Phase 3: Playwright implementation
                            local_browser era: replaced with browser_control HTTP call,
                            same filename, same function name, same signature

Internal helpers (Phase 3 implementation):
  _dom_stabilize()        — cookie/overlay removal, scroll-to-load, MathJax wait
  _detect_captcha()       — CAPTCHA presence detection
  _detect_login_wall()    — login wall / content truncation detection
  _capture_via_cdp()      — CDP PDF interception
  _extract_via_readability() — Readability.js injection via page.evaluate
  _score_pdf_links()      — PDF link heuristic scoring
"""

import logging
from typing import Optional

logger = logging.getLogger("AcquireService")

# ─────────────────────────────────────────────
# Shared Browser State
# ─────────────────────────────────────────────

_pw = None
_browser = None


# ─────────────────────────────────────────────
# Browser Lifecycle — called by api.py events
# ─────────────────────────────────────────────

async def init_browser() -> None:
    """
    Start Playwright and launch shared Chromium instance.
    Called exclusively by api.py startup_event().
    headless="new" — stealth-compatible, not legacy headless.
    """
    global _pw, _browser
    from playwright.async_api import async_playwright
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless="new",
        args=["--no-sandbox", "--disable-dev-shm-usage"]
    )


async def close_browser() -> None:
    """
    Gracefully shut down browser and Playwright instance.
    Called exclusively by api.py shutdown_event().
    """
    global _pw, _browser
    if _browser:
        try:
            await _browser.close()
        except Exception as e:
            logger.warning(f"Browser close warning: {e}")
    if _pw:
        try:
            await _pw.stop()
        except Exception as e:
            logger.warning(f"Playwright stop warning: {e}")


def get_browser():
    """
    Return shared browser instance.
    Called by acquire_with_browser() and health check in api.py.
    Never returns a context or page — those are per-acquisition only.
    """
    return _browser


# ─────────────────────────────────────────────
# Public Entry Point — frozen contract
# ─────────────────────────────────────────────

async def acquire_with_browser(
        url: str,
        context_type: str,
        trace_id: Optional[str] = None,
) -> tuple[Optional[bytes], str]:
    """
    Sole public entry point for all browser-based acquisition.
    Called only by _acquire_via_browser() in acquire_logic.py.

    Signature is frozen. When local_browser is ready, this file is
    replaced in its entirety. Function name and signature do not change.

    Stealth discipline enforced on every page creation:
      context = await browser.new_context()
      page    = await context.new_page()
      await stealth_async(page)    <- always before page.goto()
      await page.goto(url)         <- never before stealth_async()

    Context isolation enforced on every acquisition:
      new_context() is called per acquisition, never shared.
      _browser is the only shared resource.
    """
    raise NotImplementedError("acquire_with_browser: Phase 3 implementation pending")