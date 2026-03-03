# ~/TTS/my_app/acquire_service.py
# ════════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS, CONFIGURATION, CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

import os
import re
import base64
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AcquireService")

app = FastAPI(title="Acquire Service")

# --- Environment Config ---
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")

# --- Standardized Container Paths ---
STATIC_DIR = Path("/app/static")
WORKSPACE_DIR = Path("/workspace")
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
INPUT_DIR = WORKSPACE_DIR / "pdf_input"
PDF_CACHE_DIR = WORKSPACE_DIR / "pdf_cache"

# Readability.js injected into Playwright page context
READABILITY_JS = STATIC_DIR / "Readability.js"

# Session state storage — one JSON file per domain (Playwright storage_state format)
SESSION_DIR = Path("/app/sessions")

# --- Acquisition Timeouts ---
HTTP_PROBE_TIMEOUT = 15  # seconds — HEAD/range probe
PAGE_LOAD_TIMEOUT = 30000  # ms — Playwright networkidle
PDF_DOWNLOAD_TIMEOUT = 60  # seconds — PDF stream download
STEALTH_RETRY_DELAY = 2.5  # seconds — delay before stealth retry

# --- Retry Config ---
# [Fix 3] Cap on CAPTCHA unlock retry steps to prevent runaway sequences
MAX_CAPTCHA_RETRIES = 2  # max stealth retry loops inside unlock sequence
# [Fix 4A] httpx transient retry config
HTTP_RETRY_MAX = 3
HTTP_RETRY_DELAY = 1.0  # seconds base delay (multiplied by attempt number)

# --- HTTP Range Probe ---
PDF_RANGE_BYTES = "bytes=0-1023"

# --- PDF Link Heuristic — preferred content signal terms ---
# [Fix 5] Links containing these terms are prioritised during DOM extraction
PDF_LINK_PREFERRED_TERMS = {"bill", "report", "document", "paper", "publication",
                            "filing", "release", "statement", "brief", "memo"}

# --- PDF Validation Thresholds ---
MIN_PDF_BYTES = 1024

# --- Browser UA ---
CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# [Fix 1] CORS — locked to gateway and localhost only
# Tighten in production if gateway hostname differs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://gateway:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared async HTTP client — used for probes and downstream service calls
client = httpx.AsyncClient(
    timeout=HTTP_PROBE_TIMEOUT,
    follow_redirects=True,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    headers={"User-Agent": CHROME_UA}
)

# [Fix 2] Shared Playwright browser state — started at startup, reused across requests
_pw: Optional[Any] = None
_browser: Optional[Any] = None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION A — Server Lifecycle
# ════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global _pw, _browser

    # [Fix 2] Launch shared browser at startup — contexts created per request
    try:
        from playwright.async_api import async_playwright
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        logger.info("Playwright browser: started and ready (shared, context-per-request)")
    except Exception as e:
        logger.error(f"Playwright browser failed to start: {e}")

    # Verify Readability.js is present
    if READABILITY_JS.exists():
        logger.info(f"Readability.js: found at {READABILITY_JS}")
    else:
        logger.error(f"Readability.js NOT found at {READABILITY_JS} — Cat A/E will fail")

    # Ensure session directory exists
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Session store: {SESSION_DIR}")

    # Downstream service connectivity checks (non-blocking — log only)
    for name, url in [
        ("pdf-processor", PDF_SERVICE_URL),
        ("doctr", DOCTR_SERVICE_URL),
        ("convert", CONVERT_SERVICE_URL),
    ]:
        try:
            r = await client.get(f"{url}/health", timeout=5.0)
            logger.info(f"{name}: reachable ({r.status_code})")
        except Exception as e:
            logger.warning(f"{name}: not reachable at startup — {e}")

    logger.info("Acquire service started successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    global _pw, _browser
    # [Fix 2] Graceful browser + playwright teardown
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
    await client.aclose()
    logger.info("HTTP client and browser closed.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION B — Health
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Health check — verifies shared browser and Readability.js are available.
    Returns degraded status if either is missing.
    """
    checks = {}

    # [Fix 2] Check shared browser state — no launch overhead on health checks
    if _browser and _browser.is_connected():
        checks["playwright"] = "ok"
    else:
        checks["playwright"] = "error: browser not connected"

    checks["readability_js"] = "ok" if READABILITY_JS.exists() else "error: file not found"
    checks["session_dir"] = "ok" if SESSION_DIR.exists() else "error: directory missing"

    all_ok = all("error" not in str(v) for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION C — Main Acquire Endpoint
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/acquire")
async def acquire(request: dict):
    """
    Accept a URL. Detect category. Run acquisition pipeline.
    Return canonical PDF bytes, or span JSON on OCR fallback.

    Input:
      { "url": "https://...", "category": "auto" | "A" | "B" | "D" | "E" }

    Output (success):
      PDF bytes with headers:
        X-Acquire-Source: url
        X-Acquire-Method: probe|cat_a|cat_b|cat_e
        Content-Type: application/pdf

    Output (OCR fallback):
      JSON span payload from docTR with header:
        X-Acquire-Source: url
        X-Acquire-Method: ocr_fallback
        Content-Type: application/json

    Output (failure):
      HTTP 422 with dead-letter record in response body
    """
    url = request.get("url", "").strip()
    category = request.get("category", "auto").upper()

    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    logger.info(f"Acquire: {url} [category={category}]")

    # Step 0 — HTTP probe — catch direct PDF links before any browser work
    pdf_bytes, method = await _http_probe(url)
    if pdf_bytes:
        try:
            pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
            return _pdf_response(pdf_bytes, url, method)
        except _SpanJsonResponse as sjr:
            return JSONResponse(content=sjr.payload, headers={"X-Acquire-Method": "ocr_fallback"})

    # Category routing
    if category == "auto":
        category = _detect_category(url)
        logger.info(f"Auto-detected category: {category}")

    if category == "A":
        pdf_bytes, method = await _acquire_cat_a(url)
    elif category in ("B", "C"):
        pdf_bytes, method = await _acquire_cat_b(url)
    elif category == "E":
        pdf_bytes, method = await _acquire_cat_e(url)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Category {category} not supported by acquire service. "
                   f"Cat D (file upload) goes to /convert."
        )

    if not pdf_bytes:
        return _dead_letter(url, category, "All acquisition paths exhausted")

    try:
        pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
        return _pdf_response(pdf_bytes, url, method)
    except _SpanJsonResponse as sjr:
        return JSONResponse(content=sjr.payload, headers={"X-Acquire-Method": "ocr_fallback"})


# ════════════════════════════════════════════════════════════════════════════════
# SECTION D — HTTP Probe (Step 0, all categories)
# ════════════════════════════════════════════════════════════════════════════════

async def _http_probe(url: str) -> tuple[Optional[bytes], str]:
    """
    Attempt to retrieve a direct PDF via HTTP before spinning up any browser.
    Used as Step 0 for all URL-based categories.

    Strategy:
      1. HEAD request — check Content-Type header
      2. GET Range bytes=0-1023 — confirm PDF magic bytes (%PDF-)
      3. Full GET if confirmed PDF

    Returns (pdf_bytes, method) or (None, "probe_miss")
    """
    try:
        # HEAD probe
        head = await _http_get_with_retry(url, timeout=HTTP_PROBE_TIMEOUT)
        content_type = head.headers.get("content-type", "")

        if "pdf" in content_type.lower():
            logger.info(f"Probe: Content-Type signals PDF — fetching {url}")
            r = await _http_get_with_retry(url, timeout=PDF_DOWNLOAD_TIMEOUT)
            if r.status_code == 200 and _is_pdf_bytes(r.content):
                logger.info(f"Probe: direct PDF fetch success ({len(r.content) // 1024} KB)")
                return r.content, "probe_direct"

        # Range probe — detect PDF magic bytes without full download
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
        logger.info(f"Probe: timeout for {url} — falling through to browser")
    except Exception as e:
        logger.info(f"Probe: {url} — {e} — falling through to browser")

    return None, "probe_miss"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION E — Category A: Unlocked Web (HTML → PDF)
# ════════════════════════════════════════════════════════════════════════════════

async def _acquire_cat_a(url: str) -> tuple[Optional[bytes], str]:
    """
    Cat A — Unlocked web content.
    Shared HTML→PDF core reused by Cat B (post-unlock) and Cat E.

    Pipeline:
      Playwright navigate → wait networkidle → Readability extract
      → page.pdf() → canonical PDF

    [Fix 2] Uses shared _browser — creates context per request, closes on exit.
    """
    from playwright_stealth import Stealth

    logger.info(f"Cat A: {url}")

    if not _browser or not _browser.is_connected():
        logger.error("Cat A: shared browser not available")
        return None, "cat_a_no_browser"

    context = await _browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=CHROME_UA
    )
    try:
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        await page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        logger.info(f"Cat A: page loaded — {url}")

        await _dom_stabilize(page)

        # Readability extraction — clean article body
        readability_js = READABILITY_JS.read_text(encoding="utf-8")
        article_html = await page.evaluate(f"""
            () => {{
                {readability_js}
                const reader = new Readability(document.cloneNode(true));
                const article = reader.parse();
                return article ? article.content : document.documentElement.outerHTML;
            }}
        """)

        if not article_html:
            logger.warning(f"Cat A: Readability returned empty — using full page HTML")
            article_html = await page.content()

        await page.set_content(article_html, wait_until="networkidle")
        pdf_bytes = await page.pdf(
            format="A4",
            margin={"top": "20mm", "bottom": "20mm",
                    "left": "15mm", "right": "15mm"},
            print_background=False
        )

        logger.info(f"Cat A: PDF produced ({len(pdf_bytes) // 1024} KB)")
        return pdf_bytes, "cat_a"

    except Exception as e:
        logger.error(f"Cat A failed for {url}: {e}")
        return None, "cat_a_fail"
    finally:
        await context.close()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION F — Category B: Locked Web + Gov (unlock → Cat A core)
# ════════════════════════════════════════════════════════════════════════════════

async def _acquire_cat_b(url: str) -> tuple[Optional[bytes], str]:
    """
    Cat B — Paywalled, access-controlled, or gov-gated content.

    Pipeline:
      Step 1 — Unpaywall (DOI) / Internet Archive / archive.ph / 12ft.io
      Step 2 — Playwright + stealth + CDP intercept
               CAPTCHA gate: stealth retry → camoufox → session_manager → archive fallback
      Step 3 — Cat A HTML→PDF core on unlocked page

    CAPTCHA order (locked in architecture):
      stealth retry → camoufox (Cloudflare) → session_manager → archive fallback → Googlebot spoof
    """
    logger.info(f"Cat B: {url}")

    # Step 1 — Open-access bypass: Unpaywall / Archive
    doi = _extract_doi(url)
    if doi:
        pdf_bytes = await _try_unpaywall(doi)
        if pdf_bytes:
            return pdf_bytes, "cat_b_unpaywall"

    archive_url = await _try_internet_archive(url)
    if archive_url:
        logger.info(f"Cat B: Internet Archive copy found — {archive_url}")
        pdf_bytes, _ = await _acquire_cat_a(archive_url)
        if pdf_bytes:
            return pdf_bytes, "cat_b_archive"

    mirror_url = _try_mirror_bypass(url)
    if mirror_url:
        logger.info(f"Cat B: mirror bypass URL — {mirror_url}")
        pdf_bytes, _ = await _acquire_cat_a(mirror_url)
        if pdf_bytes:
            return pdf_bytes, "cat_b_mirror"

    # Step 2 — Browser with CDP PDF intercept + stealth
    pdf_bytes, method = await _browser_with_cdp_intercept(url, category="B")
    if pdf_bytes:
        return pdf_bytes, method

    # Step 3 — Cat A HTML→PDF core on (possibly locked) page
    pdf_bytes, _ = await _acquire_cat_a(url)
    if pdf_bytes:
        return pdf_bytes, "cat_b_html_fallback"

    return None, "cat_b_fail"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION G — Category E: Social / Informal (login-walled, truncated feeds)
# ════════════════════════════════════════════════════════════════════════════════

async def _acquire_cat_e(url: str) -> tuple[Optional[bytes], str]:
    """
    Cat E — Social platforms, Substack, Medium, Twitter/X, LinkedIn, forums.

    Pipeline:
      Step 0 — Internet Archive / archive.ph (attempt before any login)
      Step 1 — Browser + stealth (attempt without session first)
               Login wall / truncation detected? → session_manager(domain)
               DOM stabilization: expand collapsed, scroll-to-load, remove overlays
      Step 2 — Cat A HTML→PDF core on unlocked page
    """
    logger.info(f"Cat E: {url}")

    # Step 0 — Archive before browser
    archive_url = await _try_internet_archive(url)
    if archive_url:
        pdf_bytes, _ = await _acquire_cat_a(archive_url)
        if pdf_bytes:
            return pdf_bytes, "cat_e_archive"

    mirror_url = _try_mirror_bypass(url)
    if mirror_url:
        pdf_bytes, _ = await _acquire_cat_a(mirror_url)
        if pdf_bytes:
            return pdf_bytes, "cat_e_mirror"

    # Step 1 — Browser without session first
    pdf_bytes, method = await _browser_with_cdp_intercept(url, category="E")
    if pdf_bytes:
        return pdf_bytes, method

    # Step 1b — Session-authenticated retry
    domain = _extract_domain(url)
    session_path = _get_session_path(domain)
    if session_path.exists():
        logger.info(f"Cat E: retrying with stored session for {domain}")
        pdf_bytes, method = await _browser_with_session(url, session_path)
        if pdf_bytes:
            return pdf_bytes, "cat_e_session"

    # Step 2 — Cat A HTML→PDF core
    pdf_bytes, _ = await _acquire_cat_a(url)
    if pdf_bytes:
        return pdf_bytes, "cat_e_html_fallback"

    return None, "cat_e_fail"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION H — CDP Intercept Browser (shared Cat B / Cat E)
# ════════════════════════════════════════════════════════════════════════════════

async def _browser_with_cdp_intercept(
        url: str,
        category: str
) -> tuple[Optional[bytes], str]:
    """
    Playwright browser with:
      - playwright-stealth hardening (always on)
      - CDP Network.responseReceived for PDF detection
      - page download hook for triggered PDF downloads
      - DOM stabilization before CAPTCHA gate
      - CAPTCHA gate: stealth retry → camoufox → session_manager → archive fallback

    [Fix 2] Uses shared _browser — context created and closed per call.
    [Fix 5] PDF DOM links scored and sorted by preferred content terms.

    Returns (pdf_bytes, method) or (None, "cdp_miss").
    """
    from playwright_stealth import Stealth

    if not _browser or not _browser.is_connected():
        logger.error("CDP intercept: shared browser not available")
        return None, "cdp_no_browser"

    captured_pdf: list = []
    logger.info(f"CDP intercept browser: {url} [cat={category}]")

    context = await _browser.new_context(
        viewport={"width": 1280, "height": 900},
        user_agent=CHROME_UA,
        accept_downloads=True
    )
    try:
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        # CDP PDF interception — catch inline PDF responses before page render
        cdp = await context.new_cdp_session(page)
        await cdp.send("Network.enable")

        async def _on_response_received(event):
            resp = event.get("response", {})
            ct = resp.get("mimeType", "") or resp.get("headers", {}).get("content-type", "")
            req_url = resp.get("url", "")
            if "pdf" in ct.lower() or req_url.lower().endswith(".pdf"):
                req_id = event.get("requestId")
                logger.info(f"CDP: PDF response intercepted — {req_url}")
                try:
                    body_resp = await cdp.send("Network.getResponseBody", {"requestId": req_id})
                    body = body_resp.get("body", "")
                    b64 = body_resp.get("base64Encoded", False)
                    data = base64.b64decode(body) if b64 else body.encode()
                    if _is_pdf_bytes(data):
                        captured_pdf.append(data)
                        logger.info(f"CDP: captured {len(data) // 1024} KB")
                except Exception as e:
                    logger.warning(f"CDP: body retrieval failed — {e}")

        cdp.on("Network.responseReceived", _on_response_received)

        # Download hook — catch triggered downloads
        async def _on_download(download):
            logger.info(f"Download triggered: {download.suggested_filename}")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                await download.save_as(tmp.name)
                data = Path(tmp.name).read_bytes()
                if _is_pdf_bytes(data):
                    captured_pdf.append(data)
                    logger.info(f"Download: PDF captured ({len(data) // 1024} KB)")

        page.on("download", _on_download)

        # Navigate
        try:
            await page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        except Exception as e:
            logger.warning(f"CDP browser: navigation warning — {e}")

        # DOM stabilization — MUST fire before CAPTCHA gate (architecture rule)
        await _dom_stabilize(page)

        # [Fix 5] PDF DOM link extraction — scored by preferred content terms
        pdf_links_raw = await page.evaluate("""
            () => Array.from(
                document.querySelectorAll(
                    "a[href$='.pdf'], a[href*='/pdf/'], a[href*='=pdf'], a[href*='download']"
                )
            ).map(a => ({ href: a.href, text: (a.textContent || a.title || '').toLowerCase() }))
             .filter(x => Boolean(x.href))
        """)
        if pdf_links_raw:
            pdf_links = _score_pdf_links(pdf_links_raw)
            logger.info(f"CDP browser: {len(pdf_links)} PDF link(s) after scoring")
            for link in pdf_links[:5]:
                try:
                    r = await _http_get_with_retry(link["href"], timeout=PDF_DOWNLOAD_TIMEOUT)
                    if r.status_code == 200 and _is_pdf_bytes(r.content):
                        logger.info(
                            f"CDP browser: DOM PDF link success — {link['href']} "
                            f"(score={link['score']})"
                        )
                        return r.content, "cdp_dom_link"
                except Exception:
                    continue

        # CDP already captured a PDF during navigation?
        if captured_pdf:
            logger.info(f"CDP: PDF captured during navigation ({len(captured_pdf[0]) // 1024} KB)")
            return captured_pdf[0], "cdp_intercept"

        # CAPTCHA gate
        captcha_detected = await _detect_captcha(page)
        if captcha_detected:
            logger.warning(f"CAPTCHA detected on {url} — running unlock sequence")
            pdf_bytes = await _captcha_unlock_sequence(url, page, context, category)
            if pdf_bytes:
                return pdf_bytes, "cdp_captcha_unlock"

    except Exception as e:
        logger.error(f"CDP browser failed for {url}: {e}")
    finally:
        await context.close()

    return None, "cdp_miss"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION I — CAPTCHA Unlock Sequence
# ════════════════════════════════════════════════════════════════════════════════

async def _captcha_unlock_sequence(
        url: str,
        page,
        context,
        category: str
) -> Optional[bytes]:
    """
    CAPTCHA unlock ladder (architecture-locked order):
      1. Stealth retry          — re-navigate with delay (capped at MAX_CAPTCHA_RETRIES)
      2. camoufox               — Cloudflare-resistant Firefox profile
      3. session_manager        — use stored domain session
      4. Archive fallback       — Internet Archive / archive.ph
      5. Googlebot spoof        — retry-only UA header (last resort)

    [Fix 3] Stealth retry loop is capped at MAX_CAPTCHA_RETRIES = 2.
    Returns PDF bytes or None.

    NOTE: camoufox and session_manager are stubbed pending full implementation.
    """

    # 1 — Stealth retry (delay + re-navigate) — [Fix 3] capped loop
    logger.info(f"CAPTCHA unlock: Step 1 — stealth retry (max {MAX_CAPTCHA_RETRIES})")
    for attempt in range(MAX_CAPTCHA_RETRIES):
        try:
            await asyncio.sleep(STEALTH_RETRY_DELAY * (attempt + 1))
            await page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
            await _dom_stabilize(page)
            if not await _detect_captcha(page):
                pdf_bytes = await page.pdf(format="A4")
                if _is_pdf_bytes(pdf_bytes):
                    logger.info(f"CAPTCHA unlock: stealth retry succeeded (attempt {attempt + 1})")
                    return pdf_bytes
            logger.info(
                f"CAPTCHA unlock: stealth retry {attempt + 1}/{MAX_CAPTCHA_RETRIES} — still blocked")
        except Exception as e:
            logger.warning(f"CAPTCHA unlock: stealth retry {attempt + 1} failed — {e}")

    # 2 — camoufox (Cloudflare-resistant) [STUB]
    logger.info("CAPTCHA unlock: Step 2 — camoufox [stub]")
    # TODO: Launch camoufox Firefox browser via `camoufox` package
    #   from camoufox.async_api import AsyncCamoufox
    #   async with AsyncCamoufox(headless=True, geoip=True) as cam_browser:
    #       cam_page = await cam_browser.new_page()
    #       await cam_page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
    #       pdf_bytes = await cam_page.pdf(format="A4")
    #       if _is_pdf_bytes(pdf_bytes): return pdf_bytes
    logger.warning("CAPTCHA unlock: camoufox not yet implemented — skipping")

    # 3 — session_manager (stored domain session) [STUB]
    logger.info("CAPTCHA unlock: Step 3 — session_manager [stub]")
    # TODO: Implement _session_manager_unlock(url, context)
    #   domain = _extract_domain(url)
    #   session_path = _get_session_path(domain)
    #   if session_path.exists():
    #       await context.add_cookies([...from storage_state...])
    #       await page.goto(url, wait_until="networkidle", ...)
    #       check login wall → pdf if clear
    #   Max 1 retry, then fall through
    logger.warning("CAPTCHA unlock: session_manager not yet implemented — skipping")

    # 4 — Archive fallback
    logger.info("CAPTCHA unlock: Step 4 — archive fallback")
    archive_url = await _try_internet_archive(url)
    if archive_url:
        try:
            pdf_bytes, _ = await _acquire_cat_a(archive_url)
            if pdf_bytes:
                logger.info("CAPTCHA unlock: archive fallback succeeded")
                return pdf_bytes
        except Exception as e:
            logger.warning(f"CAPTCHA unlock: archive fallback failed — {e}")

    mirror_url = _try_mirror_bypass(url)
    if mirror_url:
        try:
            pdf_bytes, _ = await _acquire_cat_a(mirror_url)
            if pdf_bytes:
                logger.info("CAPTCHA unlock: mirror bypass succeeded")
                return pdf_bytes
        except Exception as e:
            logger.warning(f"CAPTCHA unlock: mirror bypass failed — {e}")

    # 5 — Googlebot spoof (retry-only, last resort)
    # [Fix 2] Creates new context from shared _browser
    logger.info("CAPTCHA unlock: Step 5 — Googlebot UA spoof")
    if _browser and _browser.is_connected():
        gb_context = await _browser.new_context(
            user_agent=(
                "Mozilla/5.0 (compatible; Googlebot/2.1; "
                "+http://www.google.com/bot.html)"
            )
        )
        try:
            gb_page = await gb_context.new_page()
            await gb_page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
            pdf_bytes = await gb_page.pdf(format="A4")
            if _is_pdf_bytes(pdf_bytes):
                logger.info("CAPTCHA unlock: Googlebot spoof succeeded")
                return pdf_bytes
        except Exception as e:
            logger.warning(f"CAPTCHA unlock: Googlebot spoof failed — {e}")
        finally:
            await gb_context.close()

    logger.error(f"CAPTCHA unlock: all steps exhausted for {url}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION J — Session-Authenticated Browser
# ════════════════════════════════════════════════════════════════════════════════

async def _browser_with_session(
        url: str,
        session_path: Path
) -> tuple[Optional[bytes], str]:
    """
    Launch Playwright persistent context with a stored domain session.
    Used for Cat E login-walled content after unauthenticated attempt fails.

    Session files stored as SESSION_DIR/{domain}.json (Playwright storage_state format).

    [Fix 2] Uses shared _pw (playwright instance) to launch persistent context.
    Persistent context cannot share the headless browser — needs its own launch.

    Harvesting sessions (manual, outside this service):
      playwright launch_persistent_context(user_data_dir=Chrome_profile_dir)
      → navigate to domain → let user log in → save storage_state

    TODO: Pluggable automated login hook per domain (future)
    """
    from playwright_stealth import Stealth

    logger.info(f"Session browser: {url} [session={session_path.name}]")

    if not _pw:
        logger.error("Session browser: playwright instance not available")
        return None, "session_no_playwright"

    context = await _pw.chromium.launch_persistent_context(
        user_data_dir=str(SESSION_DIR / "profiles" / session_path.stem),
        headless=True,
        storage_state=str(session_path),
        viewport={"width": 1280, "height": 900},
    )
    try:
        page = await context.new_page()
        await Stealth().apply_stealth_async(page)

        await page.goto(url, wait_until="networkidle", timeout=PAGE_LOAD_TIMEOUT)
        await _dom_stabilize(page)

        if await _detect_login_wall(page):
            logger.warning(f"Session: still locked after session load — {url}")
            return None, "session_still_locked"

        readability_js = READABILITY_JS.read_text(encoding="utf-8")
        article_html = await page.evaluate(f"""
            () => {{
                {readability_js}
                const reader = new Readability(document.cloneNode(true));
                const article = reader.parse();
                return article ? article.content : document.documentElement.outerHTML;
            }}
        """)
        await page.set_content(article_html or await page.content(), wait_until="networkidle")
        pdf_bytes = await page.pdf(
            format="A4",
            margin={"top": "20mm", "bottom": "20mm",
                    "left": "15mm", "right": "15mm"},
            print_background=False
        )

        if _is_pdf_bytes(pdf_bytes):
            logger.info(f"Session browser: PDF produced ({len(pdf_bytes) // 1024} KB)")
            return pdf_bytes, "session_authenticated"

    except Exception as e:
        logger.error(f"Session browser failed for {url}: {e}")
    finally:
        await context.close()

    return None, "session_fail"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION K — Post-Acquisition Validation
# ════════════════════════════════════════════════════════════════════════════════

async def _post_acquisition_validate(
        pdf_bytes: bytes,
        url: str,
        method: str
) -> bytes:
    """
    Universal post-acquisition validation (all categories).

    Checks:
      1. Size > 0 and valid PDF magic bytes
      2. Page count > 0
      3. Text layer presence

    On no text layer → raises _SpanJsonResponse (caller returns as JSON)
    On empty/invalid → raises HTTPException 422
    Returns validated pdf_bytes on pass.
    """
    if not pdf_bytes or not _is_pdf_bytes(pdf_bytes):
        raise HTTPException(status_code=422, detail=f"Invalid PDF produced from {url}")

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            doc.close()
            raise HTTPException(status_code=422, detail=f"Zero-page PDF from {url}")

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        page_count = doc.page_count
        doc.close()

        if not full_text.strip():
            logger.info(
                f"Validation: PDF valid but has no text layer from {url} "
                f"— returning PDF bytes for processor-side OCR"
            )
        else:
            logger.info(f"Validation: PDF ok — {page_count} pages, {len(full_text)} chars")

    except (_SpanJsonResponse, HTTPException):
        raise
    except ImportError:
        logger.warning("PyMuPDF not available — skipping deep validation")
    except Exception as e:
        logger.warning(f"Validation warning for {url}: {e}")

    return pdf_bytes


class _SpanJsonResponse(Exception):
    """Internal signal: PDF has no text layer — carry span JSON response to endpoint."""

    def __init__(self, payload: dict):
        self.payload = payload


# ════════════════════════════════════════════════════════════════════════════════
# SECTION L — docTR Routing (image-only PDF fallback)
# ════════════════════════════════════════════════════════════════════════════════

async def _route_to_doctr(pdf_bytes: bytes, source_url: str) -> dict:
    """
    Send image-only PDF to docTR container (port 8004).
    Returns span JSON payload for the extraction pipeline adapter.

    docTR endpoint contract:
      POST /process  multipart: file=<pdf_bytes>
      Response: { "pages": [...], "raw_spans": [...] }
    """
    logger.info(f"Routing to docTR: {source_url} ({len(pdf_bytes) // 1024} KB)")
    try:
        resp = await client.post(
            f"{DOCTR_SERVICE_URL}/process",
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            timeout=120.0
        )
        resp.raise_for_status()
        span_json = resp.json()
        span_json["_acquire_source"] = source_url
        span_json["_acquire_method"] = "ocr_doctr"
        logger.info(f"docTR: span JSON received for {source_url}")
        return span_json
    except Exception as e:
        logger.error(f"docTR routing failed for {source_url}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"docTR OCR service failed for {source_url}: {e}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION M — DOM Stabilization + Detection Helpers
# ════════════════════════════════════════════════════════════════════════════════

async def _dom_stabilize(page) -> None:
    """
    DOM stabilization — fires BEFORE CAPTCHA detection gate (architecture rule).

    Steps:
      1. Remove cookie banners and overlays
      2. Expand collapsed content sections
      3. Wait for MathJax render (if present)
      4. Scroll-to-load for lazy-loaded content
    """
    try:
        await page.evaluate("""
            () => {
                const selectors = [
                    '[class*="cookie"]', '[class*="banner"]', '[class*="overlay"]',
                    '[class*="modal"]',  '[class*="paywall"]', '[id*="cookie"]',
                    '[id*="banner"]',    '[id*="gdpr"]',       '[class*="popup"]'
                ];
                selectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                });
            }
        """)

        await page.evaluate("""
            () => {
                document.querySelectorAll('[aria-expanded="false"]').forEach(el => {
                    try { el.click(); } catch(e) {}
                });
                document.querySelectorAll('details:not([open])').forEach(el => {
                    el.setAttribute('open', '');
                });
            }
        """)

        await page.evaluate("""
            async () => {
                if (window.MathJax && window.MathJax.typesetPromise) {
                    await window.MathJax.typesetPromise();
                }
            }
        """)

        await page.evaluate("""
            async () => {
                for (let i = 0; i < 3; i++) {
                    window.scrollBy(0, window.innerHeight);
                    await new Promise(r => setTimeout(r, 500));
                }
                window.scrollTo(0, 0);
            }
        """)

        await asyncio.sleep(0.5)

    except Exception as e:
        logger.warning(f"DOM stabilization warning: {e}")


async def _detect_captcha(page) -> bool:
    """
    Detect CAPTCHA or challenge gate presence.
    Checks for known CAPTCHA provider signatures in DOM and page title.
    """
    try:
        return bool(await page.evaluate("""
            () => {
                const html  = document.documentElement.innerHTML.toLowerCase();
                const title = document.title.toLowerCase();
                const signals = [
                    'cf-challenge', 'cf_clearance', 'cloudflare',
                    'captcha', 'hcaptcha', 'recaptcha',
                    'please verify', 'access denied', 'just a moment',
                    'checking your browser', 'ray id'
                ];
                return signals.some(s => html.includes(s) || title.includes(s));
            }
        """))
    except Exception:
        return False


async def _detect_login_wall(page) -> bool:
    """
    Detect login wall or content truncation.
    Used by Cat E session path to confirm authentication succeeded.
    """
    try:
        return bool(await page.evaluate("""
            () => {
                const html  = document.documentElement.innerHTML.toLowerCase();
                const title = document.title.toLowerCase();
                const signals = [
                    'sign in', 'log in', 'login', 'subscribe to read',
                    'members only', 'create account', 'continue reading',
                    'this post is for paid subscribers'
                ];
                return signals.some(s => html.includes(s) || title.includes(s));
            }
        """))
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════════
# SECTION N — Open Access + Mirror Helpers
# ════════════════════════════════════════════════════════════════════════════════

async def _try_unpaywall(doi: str) -> Optional[bytes]:
    """
    Query Unpaywall API for open-access PDF of a DOI.
    Returns PDF bytes on success, None if not found or unavailable.

    Unpaywall API: https://api.unpaywall.org/v2/{doi}?email=your@email.com
    TODO: Add UNPAYWALL_EMAIL to docker-compose.acquire.yml environment block.
    """
    email = os.getenv("UNPAYWALL_EMAIL", "acquire@pipeline.local")
    try:
        resp = await _http_get_with_retry(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=10.0
        )
        if resp.status_code == 200:
            data = resp.json()
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("url_for_pdf")
            if pdf_url:
                logger.info(f"Unpaywall: open-access PDF found — {pdf_url}")
                pdf_resp = await _http_get_with_retry(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT)
                if pdf_resp.status_code == 200 and _is_pdf_bytes(pdf_resp.content):
                    return pdf_resp.content
    except Exception as e:
        logger.info(f"Unpaywall: {doi} — {e}")
    return None


async def _try_internet_archive(url: str) -> Optional[str]:
    """
    Query Internet Archive Wayback Machine for a recent snapshot URL.
    Returns snapshot URL string on success, None if not found.
    """
    try:
        resp = await _http_get_with_retry(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=8.0
        )
        if resp.status_code == 200:
            data = resp.json()
            snapshot = data.get("archived_snapshots", {}).get("closest", {})
            if snapshot.get("available") and snapshot.get("url"):
                logger.info(f"Internet Archive: snapshot found — {snapshot['url']}")
                return snapshot["url"]
    except Exception as e:
        logger.info(f"Internet Archive: {url} — {e}")
    return None


def _try_mirror_bypass(url: str) -> Optional[str]:
    """
    Return a mirror bypass URL (12ft.io) for the given URL.
    Lightweight URL transform — no network call required.
    archive.ph requires interaction, so 12ft.io is used here.
    """
    if "archive.ph" in url or "12ft.io" in url:
        return None
    return f"https://12ft.io/proxy?q={url}"


# ════════════════════════════════════════════════════════════════════════════════
# SECTION O — Category Detection + URL Utilities
# ════════════════════════════════════════════════════════════════════════════════

def _detect_category(url: str) -> str:
    """
    Auto-detect acquisition category from URL patterns.
    Cat C is absorbed into Cat B (architecture rule).

    Returns: "A" | "B" | "E"
    Default: "A" (safest — falls back gracefully)
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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION P — PDF Link Heuristic Scoring
# ════════════════════════════════════════════════════════════════════════════════

def _score_pdf_links(links: list) -> list:
    """
    [Fix 5] Score and sort PDF links extracted from page DOM.
    Links whose href or link text contains preferred content signal terms
    are ranked higher to reduce false captures.

    Preferred terms: bill, report, document, paper, publication,
                     filing, release, statement, brief, memo

    Returns sorted list (highest score first), original order preserved for ties.
    """
    scored = []
    for link in links:
        href = link.get("href", "").lower()
        text = link.get("text", "").lower()
        combined = href + " " + text
        score = sum(1 for term in PDF_LINK_PREFERRED_TERMS if term in combined)
        scored.append({**link, "score": score})

    # Stable sort: highest score first
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ════════════════════════════════════════════════════════════════════════════════
# SECTION Q — HTTP Retry Wrapper
# ════════════════════════════════════════════════════════════════════════════════

async def _http_get_with_retry(url: str, **kwargs) -> httpx.Response:
    """
    [Fix 4A] GET with exponential backoff retry on transient network failures.
    Retries on ConnectError and ReadTimeout only — not on HTTP status errors.

    Max retries: HTTP_RETRY_MAX (default 3)
    Delay: HTTP_RETRY_DELAY * attempt (1s, 2s, 3s)
    """
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_RETRY_MAX):
        try:
            return await client.get(url, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exc = e
            if attempt < HTTP_RETRY_MAX - 1:
                wait = HTTP_RETRY_DELAY * (attempt + 1)
                logger.info(
                    f"HTTP retry {attempt + 1}/{HTTP_RETRY_MAX}: {url} — {e} — waiting {wait}s")
                await asyncio.sleep(wait)
    raise last_exc


# ════════════════════════════════════════════════════════════════════════════════
# SECTION R — PDF Validation Utilities + Response Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _is_pdf_bytes(data: bytes) -> bool:
    """Check PDF magic bytes. Fast — no parsing."""
    return bool(data) and data[:5] == b"%PDF-"


def _pdf_response(pdf_bytes: bytes, url: str, method: str) -> Response:
    """Wrap PDF bytes in a standard Response with acquisition metadata headers."""
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "X-Acquire-Source": url[:500],
            "X-Acquire-Method": method,
        }
    )


def _dead_letter(url: str, category: str, reason: str) -> JSONResponse:
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
        "timestamp": datetime.utcnow().isoformat(),
    }
    logger.error(f"Dead letter: {record}")
    return JSONResponse(status_code=422, content=record)