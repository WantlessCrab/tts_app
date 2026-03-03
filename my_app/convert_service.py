# ~/TTS/my_app/convert_service.py
# ════════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS, CONFIGURATION, CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

import os
import io
import magic
import logging
import asyncio
import tempfile
import subprocess
from pathlib import Path
from functools import partial

import httpx
import mammoth
import ebooklib
from ebooklib import epub
from weasyprint import HTML as WeasyHTML
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ConvertService")

app = FastAPI(title="Convert Service")

# --- Environment Config ---
SOFFICE_PATH = os.getenv("SOFFICE_PATH", "/usr/bin/soffice")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://host.docker.internal:8001")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")

# --- Standardized Container Paths ---
WORKSPACE_DIR = Path("/workspace")
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
INPUT_DIR = WORKSPACE_DIR / "pdf_input"
PDF_CACHE_DIR = WORKSPACE_DIR / "pdf_cache"

# --- Quality Gate Thresholds ---
MIN_PDF_SIZE_BYTES = 1024  # < 1KB = empty/failed conversion
MIN_CHAR_COUNT = 50  # fewer chars = blank or near-blank
MAX_GARBLE_RATIO = 0.15  # >15% non-printable = encoding failure
MAX_GLYPH_RATIO = 0.10  # >10% replacement glyphs (□ / \ufffd)

# --- Format Routing Maps ---
OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "application/msword",  # DOC
    "application/vnd.ms-powerpoint",  # PPT
    "application/vnd.ms-excel",  # XLS
    "application/vnd.oasis.opendocument.text",  # ODT
    "application/vnd.oasis.opendocument.presentation",  # ODP
    "application/vnd.oasis.opendocument.spreadsheet",  # ODS
    "application/rtf",
    "text/rtf",
}
EBOOK_MIMES = {
    "application/epub+zip",  # EPUB
    "application/x-mobipocket-ebook",  # MOBI
    "application/vnd.amazon.ebook",  # AZW
}
OFFICE_EXTENSIONS = {".docx", ".doc", ".odt", ".pptx", ".ppt", ".odp", ".xlsx", ".xls", ".ods",
                     ".rtf"}
EBOOK_EXTENSIONS = {".epub", ".mobi", ".azw", ".azw3"}
DOCX_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}

# [Fix 1] CORS — locked to gateway and localhost only
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

# Shared HTTP client for downstream service calls
client = httpx.AsyncClient(timeout=60.0)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION A — Server Lifecycle
# ════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    # Verify LibreOffice — [Fix 4C] use asyncio.to_thread for subprocess
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [SOFFICE_PATH, "--version"],
            capture_output=True, text=True, timeout=10
        )
        logger.info(f"LibreOffice ready: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"LibreOffice not available at {SOFFICE_PATH}: {e}")

    # Verify Calibre
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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION B — Health
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """
    Health check — verifies LibreOffice and Calibre are reachable.
    Returns degraded status if either tool is unavailable.
    [Fix 4C] Subprocess calls run in thread pool via asyncio.to_thread.
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
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks
    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION C — Main Convert Endpoint
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """
    Accept an uploaded office or ebook file.
    Detect format via MIME + extension.
    Route to correct conversion path.
    Return canonical PDF bytes.

    Output priority (PDF-first hierarchy):
      1. Native LibreOffice / Calibre canonical PDF
      2. Mammoth → WeasyPrint PDF       (DOCX fallback)
      3. EPUB HTML → WeasyPrint PDF     (EPUB fallback)
      4. Docling span JSON              (last resort stub — pending layout container)
    """
    raw = await file.read()
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    logger.info(f"Received: {filename} ({len(raw) // 1024} KB)")

    mime = _detect_mime(raw)
    logger.info(f"MIME: {mime} | ext: {ext}")

    if mime in OFFICE_MIMES or ext in OFFICE_EXTENSIONS:
        pdf_bytes = await _convert_office(raw, filename, ext, mime)
    elif mime in EBOOK_MIMES or ext in EBOOK_EXTENSIONS:
        pdf_bytes = await _convert_ebook(raw, filename, ext, mime)
    else:
        logger.warning(f"Unsupported format: mime={mime} ext={ext}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: mime={mime}, ext={ext}"
        )

    logger.info(f"Conversion complete: {filename} → {len(pdf_bytes) // 1024} KB PDF")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"X-Source-File": filename}
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION D — Office Conversion Path (Route A)
# ════════════════════════════════════════════════════════════════════════════════

async def _convert_office(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str
) -> bytes:
    """
    Route A: DOCX / ODT / PPTX / XLSX / RTF / ODS

    Step 1 — LibreOffice headless → canonical PDF
    Step 2 — Quality gate
    Step 3 — On fail:
               DOCX → Mammoth → WeasyPrint
               Other → Docling fallback stub

    [Fix 4C] subprocess.run wrapped in asyncio.to_thread — non-blocking.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir()
        in_path.write_bytes(raw)

        # Step 1 — LibreOffice headless
        logger.info(f"LibreOffice: converting {filename}")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    SOFFICE_PATH,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", str(out_dir),
                    str(in_path)
                ],
                capture_output=True, text=True, timeout=120
            )
            pdf_candidates = list(out_dir.glob("*.pdf"))

            if result.returncode == 0 and pdf_candidates:
                pdf_bytes = pdf_candidates[0].read_bytes()

                # Step 2 — Quality gate
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename)
                if not issues:
                    logger.info(f"LibreOffice: quality gate passed for {filename}")
                    return pdf_bytes
                else:
                    logger.warning(f"LibreOffice: quality gate failed — {issues}")
            else:
                logger.warning(f"LibreOffice: conversion failed — {result.stderr[:300]}")

        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice: timeout on {filename}")
        except Exception as e:
            logger.error(f"LibreOffice: unexpected error — {e}")

    # Step 3 — Fallback routing
    is_docx = (ext == ".docx" or mime in DOCX_MIMES)
    if is_docx:
        logger.info(f"Fallback: Mammoth → WeasyPrint for {filename}")
        return await _mammoth_weasyprint(raw, filename)
    else:
        logger.info(f"Fallback: Docling stub for {filename} (non-DOCX office format)")
        return await _docling_fallback(raw, filename)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION E — Ebook Conversion Path (Route B)
# ════════════════════════════════════════════════════════════════════════════════

async def _convert_ebook(
        raw: bytes,
        filename: str,
        ext: str,
        mime: str
) -> bytes:
    """
    Route B: EPUB / MOBI / AZW

    Step 1 — Calibre → canonical PDF
    Step 2 — Quality gate
    Step 3 — On fail:
               EPUB → HTML extraction → WeasyPrint
               MOBI/AZW → Docling fallback stub

    [Fix 4C] subprocess.run wrapped in asyncio.to_thread — non-blocking.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / filename
        out_path = Path(tmpdir) / "output.pdf"
        in_path.write_bytes(raw)

        # Step 1 — Calibre
        logger.info(f"Calibre: converting {filename}")
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ebook-convert", str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=180
            )

            if result.returncode == 0 and out_path.exists():
                pdf_bytes = out_path.read_bytes()

                # Step 2 — Quality gate
                issues = await asyncio.to_thread(_quality_gate, pdf_bytes, filename)
                if not issues:
                    logger.info(f"Calibre: quality gate passed for {filename}")
                    return pdf_bytes
                else:
                    logger.warning(f"Calibre: quality gate failed — {issues}")
            else:
                logger.warning(f"Calibre: conversion failed — {result.stderr[:300]}")

        except subprocess.TimeoutExpired:
            logger.error(f"Calibre: timeout on {filename}")
        except Exception as e:
            logger.error(f"Calibre: unexpected error — {e}")

    # Step 3 — Fallback routing
    is_epub = (ext == ".epub" or mime == "application/epub+zip")
    if is_epub:
        logger.info(f"Fallback: EPUB HTML extraction → WeasyPrint for {filename}")
        return await _epub_html_weasyprint(raw, filename)
    else:
        logger.info(f"Fallback: Docling stub for {filename} (MOBI/AZW)")
        return await _docling_fallback(raw, filename)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION F — Fallback Renderers
# ════════════════════════════════════════════════════════════════════════════════

async def _mammoth_weasyprint(raw: bytes, filename: str) -> bytes:
    """
    DOCX → Mammoth semantic HTML → WeasyPrint → PDF
    Used when LibreOffice reflow produces unacceptable output.
    WeasyPrint render is CPU-bound — run in thread pool.
    """
    try:
        with io.BytesIO(raw) as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
            if result.messages:
                for msg in result.messages:
                    logger.info(f"Mammoth [{filename}]: {msg.message}")

        # [Fix 4C] WeasyPrint is synchronous and CPU-bound
        pdf_bytes = await asyncio.to_thread(WeasyHTML(string=html).write_pdf)
        logger.info(f"Mammoth→WeasyPrint: {filename} → {len(pdf_bytes) // 1024} KB")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Mammoth→WeasyPrint failed for {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"All DOCX conversion paths failed for {filename}: {e}"
        )


async def _epub_html_weasyprint(raw: bytes, filename: str) -> bytes:
    """
    EPUB → extract HTML content via ebooklib → WeasyPrint → PDF
    Used when Calibre PDF conversion produces unacceptable output.
    Preserves spine order from EPUB structure.
    WeasyPrint render is CPU-bound — run in thread pool.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            book = epub.read_epub(tmp_path)
            html_parts = []

            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                content = item.get_content().decode("utf-8", errors="replace")
                html_parts.append(content)

            if not html_parts:
                raise ValueError("No HTML content items found in EPUB")

            combined = "\n".join(html_parts)
            # [Fix 4C] WeasyPrint is synchronous and CPU-bound
            pdf_bytes = await asyncio.to_thread(WeasyHTML(string=combined).write_pdf)
            logger.info(f"EPUB→WeasyPrint: {filename} → {len(pdf_bytes) // 1024} KB")
            return pdf_bytes

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"EPUB→WeasyPrint failed for {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"All EPUB conversion paths failed for {filename}: {e}"
        )


async def _docling_fallback(raw: bytes, filename: str) -> bytes:
    """
    Last-resort fallback — Docling native ingestion → span JSON.
    Docling runs in its own container (port 8007, future build).
    Raises 422 until that container is live.

    When layout container is implemented:
      POST raw bytes → layout service (port 8007)
      → receive span JSON → normalize_to_raw_spans() → extraction pipeline
    """
    logger.error(
        f"All PDF conversion paths exhausted for {filename}. "
        f"Docling fallback not yet implemented (layout container pending). "
        f"Manual review required."
    )
    raise HTTPException(
        status_code=422,
        detail=(
            f"Could not produce a usable PDF from {filename}. "
            f"All conversion paths exhausted. "
            f"Docling layout fallback is pending (layout container not yet built)."
        )
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION G — Quality Gate
# ════════════════════════════════════════════════════════════════════════════════

def _quality_gate(pdf_bytes: bytes, source_name: str = "") -> list:
    """
    Run quality checks on a converted PDF. Synchronous — called via asyncio.to_thread.
    Returns a list of issue strings — empty list = pass.

    [Fix 4C] This function is synchronous (PyMuPDF is not async-safe).
    Caller must wrap with asyncio.to_thread.

    Checks:
      1. File size threshold        — catches empty / failed conversions
      2. Page count > 0             — catches zero-page outputs
      3. Character count threshold  — catches blank conversions
      4. Garbled character ratio    — catches encoding failures
      5. Replacement glyph ratio    — catches font failures (□ / U+FFFD)
      6. Page count collapse        — catches reflow failures on multi-section docs
    """
    issues = []

    # Check 1 — File size
    if len(pdf_bytes) < MIN_PDF_SIZE_BYTES:
        issues.append(f"file too small: {len(pdf_bytes)} bytes")
        return issues

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count

        # Check 2 — Page count
        if page_count == 0:
            issues.append("zero pages")
            doc.close()
            return issues

        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # Check 3 — Character count
        if len(full_text.strip()) < MIN_CHAR_COUNT:
            issues.append(f"insufficient text: {len(full_text.strip())} chars")

        # Check 4 — Garbled character ratio
        if full_text:
            non_printable = sum(
                1 for c in full_text
                if not c.isprintable() and c not in "\n\r\t "
            )
            garble_ratio = non_printable / max(len(full_text), 1)
            if garble_ratio > MAX_GARBLE_RATIO:
                issues.append(f"high garble ratio: {garble_ratio:.2%}")

        # Check 5 — Replacement glyph ratio (□ U+25A1, replacement char U+FFFD)
        if full_text:
            glyph_count = full_text.count("\u25a1") + full_text.count("\ufffd")
            glyph_ratio = glyph_count / max(len(full_text), 1)
            if glyph_ratio > MAX_GLYPH_RATIO:
                issues.append(
                    f"high replacement glyph ratio: {glyph_ratio:.2%} ({glyph_count} glyphs)"
                )

        # Check 6 — Page count collapse
        if page_count == 1 and source_name:
            ext = Path(source_name).suffix.lower()
            if ext in {".docx", ".odt", ".pptx", ".odp"}:
                if len(full_text.strip()) > 500:
                    issues.append(
                        f"possible reflow collapse: {page_count} page from {ext} "
                        f"with {len(full_text.strip())} chars"
                    )

    except ImportError:
        logger.warning("PyMuPDF not available for quality gate — skipping deep checks")
    except Exception as e:
        logger.warning(f"Quality gate error for {source_name}: {e}")

    if issues:
        logger.warning(f"Quality gate [{source_name}]: FAIL — {issues}")
    else:
        logger.info(f"Quality gate [{source_name}]: PASS")

    return issues


# ════════════════════════════════════════════════════════════════════════════════
# SECTION H — MIME Detection
# ════════════════════════════════════════════════════════════════════════════════

def _detect_mime(raw: bytes) -> str:
    """
    Detect MIME type from raw file bytes using libmagic.
    Authoritative over file extension — catches misnamed files.
    Falls back to 'application/octet-stream' on failure.
    """
    try:
        return magic.from_buffer(raw, mime=True)
    except Exception as e:
        logger.warning(f"MIME detection failed: {e}")
        return "application/octet-stream"