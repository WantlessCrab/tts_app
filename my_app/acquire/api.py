# my_app/acquire/api.py
"""
Acquire Service — FastAPI application, lifecycle, health, and /acquire endpoint.
"""

import logging
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .utils import (
    client,
    PDF_SERVICE_URL,
    DOCTR_SERVICE_URL,
    CONVERT_SERVICE_URL,
    READABILITY_JS,
    SESSION_DIR,
    _dead_letter,
)
from .validation import _post_acquisition_validate, _pdf_response
from .acquire_logic import _http_probe, _detect_category, _acquire_cat_a, _acquire_cat_b, \
    _acquire_cat_e

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AcquireService")


class AcquireRequest(BaseModel):
    url: Optional[str] = None
    query: Optional[str] = None
    category: str = "auto"
    trace_id: Optional[str] = None


app = FastAPI(title="Acquire Service")

# Browser lifecycle managed by browser_handling.py.
# api.py startup/shutdown delegates to init_browser() and close_browser().

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


# ─────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        from .browser_handling import init_browser
        await init_browser()
        logger.info("Playwright browser: started and ready (shared, context-per-request)")
    except Exception as e:
        logger.error(f"Playwright browser failed to start: {e}")

    if READABILITY_JS.exists():
        logger.info(f"Readability.js: found at {READABILITY_JS}")
    else:
        logger.error(f"Readability.js NOT found at {READABILITY_JS} — Cat A/E will fail")

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Session store: {SESSION_DIR}")

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
    try:
        from .browser_handling import close_browser
        await close_browser()
    except Exception as e:
        logger.warning(f"Browser close warning: {e}")
    await client.aclose()
    logger.info("HTTP client and browser closed.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check — verifies shared browser and Readability.js are available.
    Returns degraded status if either is missing.
    """
    checks = {}
    from .browser_handling import get_browser
    _b = get_browser()
    checks["playwright"] = "ok" if (
            _b and _b.is_connected()) else "error: browser not connected"
    checks["readability_js"] = "ok" if READABILITY_JS.exists() else "error: file not found"
    checks["session_dir"] = "ok" if SESSION_DIR.exists() else "error: directory missing"
    all_ok = all("error" not in str(v) for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


# ─────────────────────────────────────────────
# Main Acquire Endpoint
# ─────────────────────────────────────────────

@app.post("/acquire")
async def acquire(body: AcquireRequest, raw_request: Request):
    """
    Accept a URL. Detect category. Run acquisition pipeline.
    Return canonical PDF bytes.

    Input:
      { "url": "https://...", "category": "auto" | "A" | "B" | "D" | "E" }

    Output (success):
      PDF bytes — Content-Type: application/pdf
        X-Acquire-Source: url
        X-Acquire-Method: probe | cat_a | cat_b | cat_e

    Output (failure):
      HTTP 422 dead-letter JSON record
    """
    trace_id = (
            raw_request.headers.get("X-Trace-ID")
            or body.trace_id
            or str(uuid.uuid4())
    )

    if body.query and not body.url:
        raise HTTPException(
            status_code=501,
            detail="query-based acquisition not yet implemented"
        )

    url = (body.url or "").strip()
    category = body.category.upper()

    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    logger.info(f"Acquire: {url} [category={category}] [trace_id={trace_id}]")

    # Step 0 — HTTP probe: catch direct PDF links before any browser work
    pdf_bytes, method = await _http_probe(url)
    if pdf_bytes:
        pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
        return _pdf_response(pdf_bytes, url, method)

    # Category routing
    if category == "auto":
        category = _detect_category(url)
        logger.info(f"Auto-detected category: {category}")

    if category == "A":
        pdf_bytes, method = await _acquire_cat_a(url, trace_id)
    elif category in ("B", "C"):
        pdf_bytes, method = await _acquire_cat_b(url, trace_id)
    elif category == "E":
        pdf_bytes, method = await _acquire_cat_e(url, trace_id)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Category {category} not supported. Cat D (file upload) goes to /convert."
        )

    if not pdf_bytes:
        return _dead_letter(url, category, "All acquisition paths exhausted", trace_id)

    pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
    return _pdf_response(pdf_bytes, url, method)