# my_app/acquire/api.py
"""
Acquire Service — FastAPI application, lifecycle, health, and /acquire endpoint.
"""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .utils import (
    client,
    PDF_SERVICE_URL,
    DOCTR_SERVICE_URL,
    CONVERT_SERVICE_URL,
    _dead_letter,
)
from .validation import _post_acquisition_validate, _pdf_response
from .acquire_logic import _http_probe, _detect_category, _acquire_cat_a, _acquire_cat_b, \
    _acquire_cat_e
from ..service_health import build_health_response

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
    for name, url in [
        ("pdf-processor", PDF_SERVICE_URL),
        ("doctr", DOCTR_SERVICE_URL),
        ("convert", CONVERT_SERVICE_URL),
    ]:
        try:
            r = await client.get(f"{url.rstrip('/')}/health", timeout=5.0)
            logger.info(f"{name}: reachable ({r.status_code})")
        except Exception as e:
            logger.warning(f"{name}: not reachable at startup — {e}")

    logger.info("Acquire service started successfully with direct-PDF acquisition enabled.")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    checks = {"direct_pdf_probe": "ok", "browser_acquisition": "unsupported"}
    for name, url in [
        ("pdf_processor", PDF_SERVICE_URL),
        ("doctr", DOCTR_SERVICE_URL),
        ("convert", CONVERT_SERVICE_URL),
    ]:
        try:
            r = await client.get(f"{url.rstrip('/')}/health", timeout=5.0)
            checks[name] = "ok" if r.status_code == 200 else f"error: HTTP {r.status_code}"
        except Exception as exc:
            checks[name] = f"error: {exc}"
    return build_health_response(
        service="tts_acquire_service",
        role="acquire_api",
        checks=checks,
        details={
            "active_acquisition": "direct_pdf_probe",
            "pdf_service_url": PDF_SERVICE_URL,
            "doctr_service_url": DOCTR_SERVICE_URL,
            "convert_service_url": CONVERT_SERVICE_URL,
        },
        status="ok" if checks.get("direct_pdf_probe") == "ok" else None,
    )


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
            status_code=422,
            detail="query-based acquisition is not an active capability; provide a direct PDF URL or upload a file"
        )

    url = (body.url or "").strip()
    category = body.category.upper()

    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    logger.info(f"Acquire: {url} [category={category}] [trace_id={trace_id}]")

    # Step 0 — HTTP probe: catch direct PDF links before category routing
    pdf_bytes, method = await _http_probe(url)
    if pdf_bytes:
        pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
        return _pdf_response(pdf_bytes, url, method, trace_id=trace_id)

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
            detail=f"Category {category} is not supported by URL acquisition. Upload files through /api/v1/ingest/file."
        )

    if not pdf_bytes:
        return _dead_letter(url, category, "All acquisition paths exhausted", trace_id)

    pdf_bytes = await _post_acquisition_validate(pdf_bytes, url, method)
    return _pdf_response(pdf_bytes, url, method, trace_id=trace_id)