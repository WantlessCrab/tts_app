# my_app/convert/api.py
"""
Convert Service — FastAPI application, lifecycle, health, and /convert endpoint.
"""

import asyncio
import logging
import os
import subprocess
import uuid

import httpx
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .convert_logic import route_conversion
from .office_handling import set_soffice_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConvertService")

class ConvertResponse(BaseModel):
    status: str
    trace_id: str
    source_filename: str
    page_count: int
    has_text_layer: bool

class ConvertErrorResponse(BaseModel):
    status: str
    trace_id: str
    reason: str
    filename: str

app = FastAPI(title="Convert Service")

# ─────────────────────────────────────────────
# Environment Config
# ─────────────────────────────────────────────

SOFFICE_PATH = os.getenv("SOFFICE_PATH", "/usr/bin/soffice")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")

# Shared HTTP client
client = httpx.AsyncClient(timeout=60.0)

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
    # Inject soffice path into office_handling
    set_soffice_path(SOFFICE_PATH)

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [SOFFICE_PATH, "--version"],
            capture_output=True, text=True, timeout=10
        )
        logger.info(f"LibreOffice ready: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"LibreOffice not available at {SOFFICE_PATH}: {e}")

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["ebook-convert", "--version"],
            capture_output=True, text=True, timeout=10
        )
        logger.info(f"Calibre ready: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Calibre not available: {e}")

    logger.info("Convert service started successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check — verifies LibreOffice and Calibre are available.
    Returns degraded status if either tool is unavailable.
    Subprocess calls run in thread pool via asyncio.to_thread.
    """
    checks = {}

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [SOFFICE_PATH, "--version"],
            capture_output=True, text=True, timeout=10
        )
        checks["libreoffice"] = result.stdout.strip() if result.returncode == 0 else "error"
    except Exception as e:
        checks["libreoffice"] = f"error: {e}"

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["ebook-convert", "--version"],
            capture_output=True, text=True, timeout=10
        )
        checks["calibre"] = result.stdout.strip() if result.returncode == 0 else "error"
    except Exception as e:
        checks["calibre"] = f"error: {e}"

    all_ok = all("error" not in str(v) for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}


# ─────────────────────────────────────────────
# Main Convert Endpoint
# ─────────────────────────────────────────────

@app.post("/convert")
async def convert(raw_request: Request, file: UploadFile = File(...)):
    """
    Accept an uploaded office or ebook file.
    Detect format via MIME + extension.
    Route to correct conversion path.
    Return canonical PDF bytes.

    UploadFile boundary validation is enforced by _quality_gate() post-receipt,
    not at the FastAPI signature layer — honest compliance for multipart endpoints.

    Output priority (PDF-first hierarchy):
      1. Native LibreOffice / Calibre canonical PDF
      2. Mammoth → WeasyPrint PDF       (DOCX fallback)
      3. EPUB HTML → WeasyPrint PDF     (EPUB fallback)
      4. Docling span JSON              (last resort stub — pending layout container)
    """
    trace_id = (
            raw_request.headers.get("X-Trace-ID")
            or str(uuid.uuid4())
    )

    raw = await file.read()
    filename = file.filename or "upload"

    logger.info(f"Received: {filename} ({len(raw) // 1024} KB) [trace_id={trace_id}]")

    pdf_bytes = await route_conversion(raw, filename, trace_id)

    logger.info(
        f"Conversion complete: {filename} → {len(pdf_bytes) // 1024} KB PDF [trace_id={trace_id}]")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"X-Source-File": filename}
    )