# ~/TTS/my_app/audio_server.py

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS, CONFIGURATION, CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from pathlib import Path
import logging
import json
import re
import httpx
from fastapi import Query
from typing import Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioServer")

app = FastAPI(title="TTS Audio Server")

# --- Standardized Container Paths ---
STATIC_DIR = Path("/app/static")
TEMPLATES_DIR = Path("/app/templates")

# --- Standardized Shared Workspace Paths ---
WORKSPACE_DIR = Path("/workspace")
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
INPUT_DIR = WORKSPACE_DIR / "pdf_input"
OBSIDIAN_DIR = WORKSPACE_DIR / "obsidian_audio"
AUDIOBOOKS_DIR = OUTPUT_DIR / "audiobooks"
PDF_CACHE_DIR = WORKSPACE_DIR / "pdf_cache"  # Read-only access to cache directory

# --- PDF Service URL (Update port) ---
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://pdf-processor-service:8000")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

AUDIO_SOURCES = {
    "audiobooks": AUDIOBOOKS_DIR,  # /workspace/outputs/audiobooks
    "obsidian": OBSIDIAN_DIR,  # /workspace/obsidian_audio
    "standalone": OUTPUT_DIR  # /workspace/outputs (for non-audiobook files)
}
# Use lowercase default consistent with keys
DEFAULT_AUDIO_SOURCE_NAME = "audiobooks"
STALE_JOB_THRESHOLD_MINUTES = int(os.getenv('STALE_JOB_THRESHOLD_MINUTES', '30'))

# APPLICATION SETUP

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the HTTP client for calling other services
client = httpx.AsyncClient(timeout=30.0)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION A — Server Lifecycle
# ════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    try:
        # Check PDF Service connectivity (uses existing HTTP client timeout of 30.0s)
        await client.get(f"{PDF_SERVICE_URL}/docs")
        logger.info(f"Successfully connected to PDF service at {PDF_SERVICE_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to PDF service at {PDF_SERVICE_URL}: {e}")

    logger.info(f"Serving audio from defined sources: {list(AUDIO_SOURCES.keys())}")
    logger.info("Audio server started successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION B — Frontend Shell / Static UI
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def get_player_interface():
    player_html_path = TEMPLATES_DIR / "player.html"
    if not player_html_path.exists():
        logger.error(f"FATAL: player.html not found at {player_html_path}")
        raise HTTPException(status_code=500, detail="Player interface file missing")
    return FileResponse(player_html_path)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION C — Source & Asset Enumeration
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/audio_sources")
async def list_audio_sources():
    """Returns a list of available audio source names."""
    return {"sources": list(AUDIO_SOURCES.keys())}

@app.get("/api/list_audio")
async def list_audio_files(source: Optional[str] = Query(DEFAULT_AUDIO_SOURCE_NAME)):
    """Lists audio files from the specified source directory."""

    # Validate the source name
    if source not in AUDIO_SOURCES:
        logger.warning(f"Invalid source requested in list_audio: {source}")
        raise HTTPException(status_code=400,
                            detail=f"Invalid audio source specified. Valid sources: {list(AUDIO_SOURCES.keys())}")

    target_directory = AUDIO_SOURCES[source]
    logger.info(f"Listing audio files from source '{source}' at path: {target_directory}")

    files = []
    if target_directory.exists():
        try:
            # Only get WAV files directly within the target directory
            for book_dir in target_directory.iterdir():
                if book_dir.is_dir():
                    for f in book_dir.glob("chunk_*.wav"):
                        files.append({
                            "name": f.name,
                            "path": str(f),
                            "book_id": book_dir.name,
                            "size_bytes": f.stat().st_size,
                            "type": source
                        })
        except Exception as e:
            logger.error(f"Error scanning directory {target_directory}: {e}")
            # Don't raise HTTPException here, just return empty list or partial results

    return {"files": files, "source": source}


@app.get("/api/available_pdfs")
async def list_available_pdfs():
    """List PDFs available for processing (Functionality retained from baseline)"""
    pdf_input_dir = Path("/workspace/pdf_input")
    pdfs = []
    if pdf_input_dir.exists():
        for pdf_file in pdf_input_dir.glob("*.pdf"):
            pdfs.append({
                "filename": pdf_file.name,
                "size_bytes": pdf_file.stat().st_size,
                "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2)
            })
    return {"available_pdfs": pdfs}


# ════════════════════════════════════════════════════════════════════════════════
# SECTION D — Audiobook Status APIs & Metadata
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/audiobooks")
async def list_audiobooks():
    """Lists all available audiobooks with their processing status"""
    books = []
    if not AUDIOBOOKS_DIR.exists():
        return {"audiobooks": books}

    for book_dir in AUDIOBOOKS_DIR.iterdir():
        if book_dir.is_dir():
            # 1.4 Sanitization Consistency: book_dir.name is already sanitized by process.py
            manifest_path = book_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    # NOTE: We do not merge citation data here (1.1 Response Simplification)
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        total_chunks = manifest.get('total_chunks', 0)
                        ready_chunks = len(manifest.get('ready_chunks', []))

                        # Use citation_ready.json file existence as proxy for highlighting availability
                        safe_book_id = book_dir.name
                        citation_filename = safe_book_id + '_citation_ready.json'
                        highlighting_ready = (PDF_CACHE_DIR / citation_filename).exists()

                        books.append({
                            "book_id": book_dir.name,
                            "title": manifest['metadata'].get('title', book_dir.name),
                            "author": manifest['metadata'].get('author', 'Unknown'),
                            "source_file": manifest['metadata'].get('source_filename', ''),
                            "total_chunks": total_chunks,
                            "ready_chunks": ready_chunks,
                            "is_complete": ready_chunks == total_chunks and total_chunks > 0,
                            "highlighting_ready": highlighting_ready  # New status field
                        })
                except Exception as e:
                    logger.error(f"Failed to read manifest for {book_dir.name}: {e}")
    return {"audiobooks": books}

@app.get("/api/audiobook/{book_id}/status")
async def get_audiobook_status(book_id: str):
    """
    Get detailed status with stale job detection.
    """
    safe_book_id = book_id
    live_manifest_path = AUDIOBOOKS_DIR / safe_book_id / "manifest.json"

    if not live_manifest_path.exists():
        logger.error(f"Live manifest not found at {live_manifest_path}")
        raise HTTPException(status_code=404,
                            detail=f"Audiobook '{book_id}' (live manifest) not found")

    try:
        # Load manifest
        with open(live_manifest_path, 'r') as f:
            live_manifest = json.load(f)

        # SOA-COMPLIANT: Graceful Degradation (Dict Validation)
        response_data = live_manifest

        # Infer missing fields
        total = response_data.get('total_chunks', 0)
        ready = len(response_data.get('ready_chunks', []))

        if 'processing_status' not in response_data:
            if total == 0:
                response_data['processing_status'] = 'processing_started'
            elif ready == total and total > 0:
                response_data['processing_status'] = 'stage_3_complete'
            elif ready > 0:
                response_data['processing_status'] = 'stage_3_partial'
            else:
                response_data['processing_status'] = 'unknown'

        # Ensure defaults
        response_data.setdefault('trace_id', 'N/A')
        response_data.setdefault('metadata', {})

        # Calculate progress
        total_chunks = response_data.get('total_chunks', 0)
        ready_chunks_list = response_data.get('ready_chunks', [])
        progress = (len(ready_chunks_list) / total_chunks) * 100 if total_chunks > 0 else 0

        response_data['progress_percentage'] = round(progress, 1)
        response_data['is_complete'] = len(ready_chunks_list) == total_chunks and total_chunks > 0

        # ✅ NEW: Stale job detection (Task 3)
        is_stale = False
        if response_data.get('processing_status') in ['processing_started', 'stage_1_complete',
                                                      'stage_2_complete', 'stage_3_started',
                                                      'stage_3_partial']:
            try:
                manifest_mtime = live_manifest_path.stat().st_mtime
                current_time = datetime.now().timestamp()
                minutes_since_update = (current_time - manifest_mtime) / 60

                if minutes_since_update > STALE_JOB_THRESHOLD_MINUTES:
                    is_stale = True
                    logger.warning(
                        f"Stale job detected for {book_id}: {minutes_since_update:.1f} minutes since last update")

            except Exception as e:
                logger.warning(f"Failed to check staleness for {book_id}: {e}")

        response_data['is_stale'] = is_stale
        response_data['stale_threshold_minutes'] = STALE_JOB_THRESHOLD_MINUTES

        # Check citation file (legacy)
        citation_filename = safe_book_id + '_citation_ready.json'
        citation_path = PDF_CACHE_DIR / citation_filename
        response_data['highlighting_ready'] = citation_path.exists()

        # Check UI sentences file (new UI contract)
        ui_sentences_filename = safe_book_id + '_ui_sentences.json'
        ui_sentences_path = PDF_CACHE_DIR / ui_sentences_filename
        response_data['ui_ready'] = ui_sentences_path.exists()

        return response_data

    except Exception as e:
        logger.error(f"Error reading live manifest for {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read audiobook data")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION E — UI Sentence & Citation APIs
# ════════════════════════════════════════════════════════════════════════════════
@app.get("/api/audiobook/{book_id}/ui_sentences")
async def get_audiobook_ui_sentences(book_id: str):
    """
    Serves pre-resolved UI sentence data from ui_sentences.json.

    This is the SOLE source of truth for frontend UI concerns:
    - Sentence timing
    - Pre-resolved geometry per page
    - Page turn markers
    - Sentence-to-page mapping

    Frontend MUST use this instead of /coordinates + /chunks for UI rendering.

    Returns the file byte-exact via FileResponse (no re-serialization).
    """
    safe_book_id = book_id

    ui_sentences_filename = f"{safe_book_id}_ui_sentences.json"
    ui_sentences_path = PDF_CACHE_DIR / ui_sentences_filename

    if not ui_sentences_path.exists():
        for f in PDF_CACHE_DIR.glob(f"*{safe_book_id}*_ui_sentences.json"):
            ui_sentences_path = f
            break
        else:
            logger.warning(f"UI sentences data not found for: {book_id}")
            raise HTTPException(
                status_code=404,
                detail="UI sentence data not available. Book may still be processing."
            )

    logger.info(f"Serving UI sentences file for: {book_id}")

    return FileResponse(
        path=ui_sentences_path,
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/api/audiobook/{book_id}/semantic")
async def get_audiobook_semantic(book_id: str):
    """
    Serves semantic.json for frontend semantic lookups.

    This endpoint exposes content-level metadata (e.g. cleaned_text)
    required for advanced frontend rendering policies such as
    progressive sentence highlighting.

    Semantic data remains the authoritative source for:
      - span text
      - content length
      - semantic roles

    Frontend MUST treat this as read-only.
    """

    safe_book_id = book_id

    semantic_filename = f"{safe_book_id}_semantic.json"
    semantic_path = PDF_CACHE_DIR / semantic_filename

    if not semantic_path.exists():
        for f in PDF_CACHE_DIR.glob(f"*{safe_book_id}*_semantic.json"):
            semantic_path = f
            break
        else:
            logger.warning(f"Semantic data not found for: {book_id}")
            raise HTTPException(
                status_code=404,
                detail="Semantic data not available. Book may still be processing."
            )

    logger.info(f"Serving semantic.json for: {book_id}")

    return FileResponse(
        path=semantic_path,
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=3600"
        }
    )

# ════════════════════════════════════════════════════════════════════════════════
# SECTION F — Audio Chunk Serving
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/audiobook/{book_id}/play/{chunk_filename}")
async def serve_audiobook_chunk(book_id: str, chunk_filename: str):
    """Serve a specific audio chunk from an audiobook (Functionality retained from baseline)"""
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    safe_filename = re.sub(r'[^\w\-\.]', '', chunk_filename).strip()

    if not safe_filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    file_path = AUDIOBOOKS_DIR / safe_book_id / safe_filename

    if not file_path.exists():
        logger.warning(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Audio chunk not found")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION G — PDF Proxy APIs
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/api/pdf/{pdf_filename}")
async def proxy_serve_pdf(pdf_filename: str):
    """
    Proxies request for source PDF document to the pdf-service.
    (2.2 Error Handling: Added Timeout Error handling)
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/document/{safe_filename}"
        logger.info(f"Proxying PDF request via GET: {api_url}")

        response = await client.get(api_url, timeout=30.0)
        response.raise_for_status()

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/pdf"),
            headers={
                k: v for k, v in response.headers.items()
                if k.lower() in [
                    'content-disposition', 'content-length', 'etag',
                    'accept-ranges', 'last-modified', 'cache-control'
                ]
            }
        )

    except httpx.TimeoutException:  # <-- CORRECTED
        logger.error(
            f"Error proxying PDF document: Proxy connection timed out to {PDF_SERVICE_URL}")
        raise HTTPException(status_code=504, detail="Gateway Timeout: PDF service did not respond.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error proxying PDF from pdf-service: {e.response.status_code}")
        try:
            detail = e.response.json().get("detail", "Failed to retrieve PDF")
        except:
            detail = f"Failed to retrieve PDF (HTTP {e.response.status_code})"
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error(f"Error proxying PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF service not available")


@app.post("/api/v1/process/{pdf_filename}")
async def start_pdf_processing(pdf_filename: str):
    """
    Proxy PDF processing request to pdf-processor service.
    Delegates full pipeline execution to the processor (SOA compliance).
    """
    safe_filename = re.sub(r'[^\w\s\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/process/{safe_filename}"
        logger.info(f"Proxying process request: {api_url}")

        response = await client.post(api_url)
        response.raise_for_status()

        return response.json()

    except httpx.TimeoutException:
        logger.error(f"Process request timed out for {safe_filename}")
        raise HTTPException(status_code=504, detail="Gateway Timeout: PDF service did not respond.")

    except httpx.HTTPStatusError as e:
        logger.error(f"Process request failed: {e.response.status_code}")
        try:
            detail = e.response.json().get("detail", "Processing failed")
        except Exception:
            detail = f"Processing failed (HTTP {e.response.status_code})"
        raise HTTPException(status_code=e.response.status_code, detail=detail)

    except Exception as e:
        logger.error(f"Process request error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate processing")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION H — Processing Control APIs
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/api/retry/{book_id}")
async def retry_processing(book_id: str, force_rebuild: bool = False):
    """
    Proxy retry request to pdf-processor service.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/retry/{safe_book_id}"

        # Pass force_rebuild flag as a query parameter
        response = await client.post(api_url, params={"force_rebuild": force_rebuild})
        response.raise_for_status()

        return response.json()

    except httpx.TimeoutException:
        logger.error(f"Retry request timed out for {safe_book_id}")
        raise HTTPException(status_code=504, detail="Gateway Timeout: PDF service did not respond.")

    except httpx.HTTPStatusError as e:
        logger.error(f"Retry request failed: {e.response.status_code}")
        detail = e.response.json().get("detail", "Retry failed")
        raise HTTPException(status_code=e.response.status_code, detail=detail)

    except Exception as e:
        logger.error(f"Retry request error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate retry")


@app.post("/api/v1/rebuild_selective/{book_id}")
async def rebuild_selective(
        book_id: str,
        chunk_ids: Optional[str] = None,
        pages: Optional[str] = None,
):
    """
    Proxy selective rebuild request to pdf-processor service.
    Full Stages 1+2, targeted Stage 3 audio generation.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/rebuild_selective/{safe_book_id}"

        params = {}
        if chunk_ids:
            params["chunk_ids"] = chunk_ids
        if pages:
            params["pages"] = pages

        response = await client.post(api_url, params=params)
        response.raise_for_status()

        return response.json()

    except httpx.TimeoutException:
        logger.error(f"Selective rebuild timed out for {safe_book_id}")
        raise HTTPException(status_code=504, detail="Gateway Timeout: PDF service did not respond.")

    except httpx.HTTPStatusError as e:
        logger.error(f"Selective rebuild failed: {e.response.status_code}")
        try:
            detail = e.response.json().get("detail", "Selective rebuild failed")
        except Exception:
            detail = f"Selective rebuild failed (HTTP {e.response.status_code})"
        raise HTTPException(status_code=e.response.status_code, detail=detail)

    except Exception as e:
        logger.error(f"Selective rebuild error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate selective rebuild")