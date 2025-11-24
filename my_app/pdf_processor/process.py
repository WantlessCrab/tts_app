# ~/TTS/my_app/pdf_processor/process.py
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import sys
import json
from pathlib import Path
import logging
import re  # For sentence splitting
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


# --- Pydantic Schema Definition (REPLACE EXISTING) ---
class ReadyChunkSchema(BaseModel):
    chunk_id: int
    filename: str
    page: int
    text_snippet: str
    start_time: float
    duration_seconds: float
    end_time: float
    sentences: List[dict]


class ManifestSchema(BaseModel):
    # Core Metadata
    metadata: dict = Field(...,
                           description="Original PDF metadata (title, author, source_filename)")
    book_id: str
    trace_id: str

    # State and Progress
    processing_status: Optional[str] = Field(None,  # ✅ FIXED: Must be Optional
                                             description="Current state: processing_started, stage_3_complete, failed, etc.")
    total_chunks: int = Field(0, description="Total chunks calculated in Stage 2.")
    ready_chunks: List[ReadyChunkSchema] = Field([], description="List of completed audio chunks.")

    # Error/Recovery
    error_message: Optional[str] = Field(None,
                                         description="Details if processing_status is 'failed' or 'interrupted'.")


# --- END Pydantic Schema Definition ---

# ========================================
# Configuration & Setup
# ========================================

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFProcessorService")

# Define paths relative to the /workspace
BASE_DIR = Path("/workspace")
CACHE_DIR = BASE_DIR / "pdf_cache"
INPUT_DIR = BASE_DIR / "pdf_input"
OUTPUT_DIR = BASE_DIR / "outputs" / "audiobooks"
CACHE_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Service Setup ---
app = FastAPI(title="PDF Processing Service")

# Create a persistent HTTP client for communicating with tts-service
client = httpx.AsyncClient(timeout=300.0)
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8000/api/tts")
TTS_MODEL_NAME = TTS_SERVICE_URL.replace("/api/tts", "/")
# --- Concurrency Configuration ---
TTS_MAX_CONCURRENT_REQUESTS = int(os.getenv('TTS_MAX_CONCURRENT_REQUESTS', '10'))
# Global Semaphore limits concurrent calls to the TTS service across all jobs
TTS_SEMAPHORE = asyncio.Semaphore(TTS_MAX_CONCURRENT_REQUESTS)

@app.on_event("startup")
async def startup_event():
    """Validate service configuration and connectivity on startup."""
    TTS_SERVICE_ROOT = TTS_SERVICE_URL.replace("/api/tts", "/")

    # Environment validation (Phase 2.5)
    if not TTS_SERVICE_URL or TTS_SERVICE_URL == "http://tts-service:8000/api/tts":
        logger.warning(
            "TTS_SERVICE_URL not configured or using default. Verify environment variable.")

    try:
        response = await client.get(TTS_SERVICE_ROOT)
        response.raise_for_status()
        logger.info(f"Successfully connected to TTS service at {TTS_SERVICE_ROOT}")
    except Exception as e:
        logger.error(f"Failed to connect to TTS service at {TTS_SERVICE_ROOT}: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ========================================
# API Endpoints
# ========================================

@app.post("/api/v1/process/{pdf_filename}")
async def start_pdf_processing(pdf_filename: str, background_tasks: BackgroundTasks):
    """
    Triggers the full PDF-to-Audio pipeline in the background.
    """
    safe_filename = re.sub(r'[^\w\s\-\.]', '', pdf_filename).strip()
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    pdf_path = INPUT_DIR / safe_filename
    if not pdf_path.exists():
        logger.warning(f"Process request failed: File not found at {pdf_path}")
        raise HTTPException(status_code=404, detail="PDF not found in input directory")

    # --- 1. Generate IDs and Paths ---
    book_id = derive_book_id(pdf_path.stem)
    trace_id = str(uuid.uuid4())
    audio_dir = OUTPUT_DIR / book_id
    manifest_path = audio_dir / "manifest.json"

    # 2. Create directory and initial manifest stub
    audio_dir.mkdir(parents=True, exist_ok=True)

    stub_manifest = {
        "metadata": {"source_filename": safe_filename},
        "book_id": book_id,
        "trace_id": trace_id,
        "processing_status": "processing_started",
        "total_chunks": 0,
        "ready_chunks": []
    }

    try:
        # Use validation helper for initial write
        validate_and_write_manifest(manifest_path, stub_manifest, trace_id, logger)
        logger.info(f"[{trace_id}] Manifest stub created for {book_id}. Job accepted.")
    except Exception as e:
        logger.error(f"[{trace_id}] CRITICAL: Failed to create manifest stub: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize processing state.")

    # 3. Add task with trace_id and default force_rebuild=False
    background_tasks.add_task(run_full_pipeline, safe_filename, book_id, trace_id, False)

    logger.info(f"[{trace_id}] Accepted job for {safe_filename}. Processing started in background.")
    return {"status": "processing_started", "book_id": book_id, "trace_id": trace_id}


async def run_full_pipeline(pdf_filename: str, book_id: str, trace_id: str,
                            force_rebuild: bool = False):
    """
    Full background processing chain with cache validation, state updates, and error capture.
    Handles the execution flow and manages state transitions (Stage 1 -> 2 -> 3).
    """
    logger.info(f"[{trace_id}] Pipeline started for: {pdf_filename}")

    pdf_path = INPUT_DIR / pdf_filename
    raw_cache_file_name = pdf_path.stem + "_raw.json"
    raw_cache_path = CACHE_DIR / raw_cache_file_name

    citation_path = None
    manifest_path = OUTPUT_DIR / book_id / "manifest.json"

    # Use canonical ID for citation path calculation
    citation_filename = book_id + '_citation_ready.json'
    potential_citation_path = CACHE_DIR / citation_filename

    # --- Cache Freshness Logic (No Abbreviation) ---
    citation_found = potential_citation_path.exists()
    pdf_is_newer = False

    # Check 1: Force rebuild bypasses cache and Mtime check entirely
    if force_rebuild:
        logger.info(f"[{trace_id}] Force rebuild requested. Ignoring cache and forcing Stage 1/2.")
        citation_found = False  # Treat as cache miss

    elif citation_found:
        # Check 2: Cache Freshness Check (mtime)
        try:
            pdf_mtime = pdf_path.stat().st_mtime
            cache_mtime = potential_citation_path.stat().st_mtime
            if pdf_mtime > cache_mtime:
                pdf_is_newer = True
                logger.warning(
                    f"[{trace_id}] PDF is newer than cache. Forcing rebuild (Stage 1 & 2).")
        except Exception as e:
            logger.warning(f"[{trace_id}] Failed to check mtime for {pdf_filename}: {e}")

    # --- Pipeline Execution: Central Try Block for Error Capture ---
    try:
        if citation_found and not pdf_is_newer and not force_rebuild:
            # CACHE HIT PATH
            logger.info(
                f"[{trace_id}] Citation cache found, skipping Stage 1 & 2. Path: {potential_citation_path}")
            citation_path = potential_citation_path

            # ✅ VALIDATION CALL #1: CACHE HIT PATH MANIFEST UPDATE
            try:
                # Load metadata and total_chunks from cache files
                metadata = {}
                if raw_cache_path.exists():
                    with open(raw_cache_path, 'r', encoding='utf-8') as f_raw:
                        raw_data = json.load(f_raw)
                        metadata = raw_data['metadata']

                with open(citation_path, 'r', encoding='utf-8') as f_cite:
                    citation_data = json.load(f_cite)

                # Load existing manifest stub
                with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                    manifest = json.load(f_manifest)

                # Update manifest state and total_chunks
                manifest['metadata'] = metadata
                manifest['total_chunks'] = citation_data['processing']['total_chunks']
                manifest['processing_status'] = "stage_2_complete"

                validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                logger.info(
                    f"[{trace_id}] Manifest stub updated via cache hit (total chunks: {manifest['total_chunks']}).")

            except Exception as e:
                logger.error(f"[{trace_id}] Error updating manifest on cache hit: {e}. HALTING.")
                raise  # Re-raise to trigger the main exception handler below

        else:
            # FRESH RUN PATH

            # Stage 1: Process PDF
            logger.info(f"[{trace_id}] Running Stage 1...")
            raw_cache_path = process_pdf(pdf_filename)

            # ✅ VALIDATION CALL #2: Update manifest state after Stage 1 complete
            with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                manifest = json.load(f_manifest)
            manifest['processing_status'] = "stage_1_complete"
            validate_and_write_manifest(manifest_path, manifest, trace_id, logger)

            if not raw_cache_path:
                raise RuntimeError("Stage 1 failed to produce raw cache.")

            # Stage 2: Prepare Chunks
            logger.info(f"[{trace_id}] Running Stage 2...")
            citation_path = prepare_tts_chunks_with_citations(raw_cache_path)

            if not citation_path:
                raise RuntimeError("Stage 2 failed to produce citation file.")

            # ✅ VALIDATION CALL #3: Manifest Update after successful Stage 2 (total chunks)
            with open(citation_path, 'r', encoding='utf-8') as f_cite:
                citation_data = json.load(f_cite)

            with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                manifest = json.load(f_manifest)

            manifest['metadata'] = citation_data['metadata']
            manifest['total_chunks'] = citation_data['processing']['total_chunks']
            manifest['processing_status'] = "stage_2_complete"

            validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
            logger.info(
                f"[{trace_id}] Manifest updated after Stage 2 (total chunks: {manifest['total_chunks']}).")

        # Stage 3: Generate Audio
        if citation_path:
            logger.info(f"[{trace_id}] Proceeding to Stage 3 (Audio Generation)...")

            # ✅ EDIT 1: Write stage_3_started Status (CRITICAL FIX)
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                    manifest = json.load(f_manifest)

                manifest['processing_status'] = "stage_3_started"
                validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                logger.info(f"[{trace_id}] Manifest updated to stage_3_started.")

                # Now call generate_audio_streaming
                await generate_audio_streaming(citation_path, book_id, trace_id, manifest_path)

            except Exception as e:
                # Catch any failure during the transition/generation
                logger.error(
                    f"[{trace_id}] Failed during STAGE 2->3 transition or generation call: {e}")
                raise  # Outer block handles 'failed' write

            # ✅ EDIT 2: VALIDATION CALL #4 - Final State Update (Atomic Check)
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                    manifest = json.load(f_manifest)

                if manifest.get('processing_status') not in ["failed", "stage_3_partial"]:

                    completed = len(manifest['ready_chunks'])
                    total = manifest['total_chunks']

                    # Atomic check: Only mark complete if ALL chunks succeeded
                    if completed == total:
                        manifest['processing_status'] = "stage_3_complete"
                        logger.info(
                            f"[{trace_id}] All chunks complete. Marking job as stage_3_complete.")
                    else:
                        manifest[
                            'processing_status'] = "stage_3_partial"  # Mark as partial completion
                        logger.warning(
                            f"[{trace_id}] Partial completion: {completed}/{total} chunks. Marking as stage_3_partial.")

                    validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                    logger.info(f"[{trace_id}] Pipeline FINISHED for: {pdf_filename}")

                else:
                    logger.warning(
                        f"[{trace_id}] Pipeline exited in state: {manifest.get('processing_status')}")

            except Exception as e:
                logger.error(f"[{trace_id}] Failed to write final status: {e}")

        else:
            logger.error(
                f"[{trace_id}] Pipeline HALTED before Stage 3 due to missing citation path.")


    except Exception as e:
        # ✅ VALIDATION CALL #5: CENTRAL ERROR PATH MANIFEST UPDATE
        logger.error(f"[{trace_id}] Pipeline HALTED due to critical error: {e}", exc_info=True)

        # Update manifest to 'failed' state
        try:
            if manifest_path.exists():
                # Load existing manifest to preserve 'ready_chunks' and other fields
                with open(manifest_path, 'r', encoding='utf-8') as f_manifest:
                    manifest = json.load(f_manifest)

                manifest['processing_status'] = "failed"
                manifest[
                    'error_message'] = f"Pipeline failed during setup: {e.__class__.__name__}: {str(e)}"

                validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                logger.error(f"[{trace_id}] Wrote 'failed' status to manifest.")

        except Exception as update_e:
            logger.error(f"[{trace_id}] Failed to write final 'failed' status: {update_e}")
        return  # Exit pipeline function

# ADD NEW ENDPOINT (Place near other API endpoints in process.py)
@app.post("/api/v1/retry/{book_id}")
async def retry_processing(book_id: str, background_tasks: BackgroundTasks,
                           force_rebuild: bool = False):
    """
    Retry or resume processing for a failed/partial job.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    manifest_path = OUTPUT_DIR / safe_book_id / "manifest.json"

    if not manifest_path.exists():
        logger.warning(f"Retry request for non-existent book: {safe_book_id}")
        raise HTTPException(status_code=404, detail="Book not found")

    # Load manifest to check status
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read manifest for retry: {e}")
        raise HTTPException(status_code=500, detail="Failed to read manifest")

    current_status = manifest.get('processing_status', 'unknown')

    # Only allow retry for failed or partial jobs
    if current_status not in ['failed', 'stage_3_partial']:
        logger.warning(f"Retry rejected for book {safe_book_id} with status: {current_status}")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status '{current_status}'. Only 'failed' or 'stage_3_partial' jobs can be retried."
        )

    # Get source filename and trace_id from manifest
    source_filename = manifest['metadata'].get('source_filename')
    old_trace_id = manifest.get('trace_id', 'N/A')

    if not source_filename:
        raise HTTPException(status_code=400, detail="Source filename not found in manifest")

    # Generate new trace_id for retry
    new_trace_id = str(uuid.uuid4())

    # Clear cache if force_rebuild requested
    if force_rebuild:
        logger.info(f"[{new_trace_id}] Force rebuild requested. Clearing cache for {safe_book_id}")

        # Delete cache files
        raw_cache = CACHE_DIR / f"{safe_book_id}_raw.json"
        citation_cache = CACHE_DIR / f"{safe_book_id}_citation_ready.json"

        if raw_cache.exists():
            raw_cache.unlink()
            logger.info(f"[{new_trace_id}] Deleted raw cache: {raw_cache}")

        if citation_cache.exists():
            citation_cache.unlink()
            logger.info(f"[{new_trace_id}] Deleted citation cache: {citation_cache}")

        # Clear ready_chunks to restart from beginning
        manifest['ready_chunks'] = []

    # Update manifest for retry
    manifest['processing_status'] = 'processing_started'
    manifest['trace_id'] = new_trace_id
    manifest['error_message'] = None  # Clear previous error

    try:
        validate_and_write_manifest(manifest_path, manifest, new_trace_id, logger)
    except Exception as e:
        logger.error(f"[{new_trace_id}] Failed to update manifest for retry: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize retry")

    # Launch pipeline with force_rebuild flag
    background_tasks.add_task(run_full_pipeline, source_filename, safe_book_id, new_trace_id,
                              force_rebuild)

    logger.info(
        f"[{new_trace_id}] Retry initiated for {safe_book_id} (old trace_id: {old_trace_id}, force_rebuild: {force_rebuild})")

    return {
        "status": "retry_started",
        "book_id": safe_book_id,
        "trace_id": new_trace_id,
        "previous_trace_id": old_trace_id,
        "force_rebuild": force_rebuild,
        "mode": "full_rebuild" if force_rebuild else "resume"
    }

@app.get("/api/v1/citation/{book_id}")
async def get_citation(book_id: str, timestamp: float = 0.0):
    """
    Gets citation information for a specific timestamp in an audiobook.
    Returns structural indices (span_start_index/span_end_index) for frontend resolution.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    safe_book_id_sanitized = safe_book_id.replace(' ', '_')
    citation_filename = safe_book_id_sanitized + '_citation_ready.json'
    citation_path = CACHE_DIR / citation_filename

    if not citation_path.exists():
        # Fallback: scan for partial match
        found_path = None
        for f in CACHE_DIR.glob(f"*{safe_book_id_sanitized}*citation_ready.json"):
            found_path = f
            break
        if not found_path:
            logger.warning(f"Citation file not found for book_id: {safe_book_id}")
            raise HTTPException(status_code=404, detail="Citation data not available")
        citation_path = found_path

    # Phase 2.1: Pass path to helper, not loaded data
    citation_data = get_citation_at_timestamp(citation_path, timestamp_seconds=timestamp)

    if not citation_data:
        raise HTTPException(status_code=404, detail=f"No citation found for timestamp {timestamp}")

    return citation_data


@app.get("/api/v1/document/{pdf_filename}")
async def serve_pdf_document(pdf_filename: str):
    """
    Serves the original source PDF document from the input directory.
    """
    safe_filename = re.sub(r'[^\w\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    file_path = INPUT_DIR / safe_filename

    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Document request failed: File not found at {file_path}")
        raise HTTPException(status_code=404, detail="PDF document not found")

    file_size = file_path.stat().st_size
    if file_size > 100 * 1024 * 1024:  # 100MB
        logger.error(f"PDF exceeds size limit: {file_size} bytes")
        raise HTTPException(status_code=413, detail="PDF file too large")

    logger.info(f"Serving source PDF: {file_path} ({file_size} bytes)")
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename=\"{safe_filename}\"",
            "Cache-Control": "public, max-age=3600",
            "Accept-Ranges": "bytes"
        }
    )


# ========================================
# Core Processing Pipeline
# ========================================
async def generate_single_chunk(chunk: dict, book_id: str, trace_id: str, manifest_path: Path,
                                logger):
    """
    Handles audio generation, file I/O, and manifest update for a single chunk.
    Uses the global semaphore to limit concurrent TTS requests and includes backpressure.
    """
    chunk_id = chunk['chunk_id']
    page = chunk['page']
    audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav"
    audio_dir = OUTPUT_DIR / book_id
    audio_path = audio_dir / audio_filename

    # Use global semaphore to limit concurrency (CRITICAL)
    async with TTS_SEMAPHORE:

        # 1. Skip if already processed (Resumption logic)
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            logger.error(f"[{trace_id}] Cannot read manifest for chunk {chunk_id}. Aborting chunk.")
            return False

        if audio_path.exists() or any(c['chunk_id'] == chunk_id for c in manifest['ready_chunks']):
            logger.debug(f"[{trace_id}] Skipping chunk {chunk_id} (already exists)")
            return True  # Successfully skipped

        # --- Generate audio via TTS API (Backpressure Target) ---
        try:
            # Retry loop for rate limits (HTTP 429) - Implement Backpressure
            MAX_RETRIES = 3
            RETRY_DELAY = 2  # Seconds

            for attempt in range(MAX_RETRIES):
                logger.info(
                    f"[{trace_id}] Chunk {chunk_id}: Requesting audio (Attempt {attempt + 1})...")

                params = {
                    "text": chunk['text'], "speaker_id": "", "style_wav": "", "language_id": "",
                }

                response = await client.post(TTS_SERVICE_URL, params=params, timeout=300.0)

                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY * (2 ** attempt)
                        logger.warning(
                            f"[{trace_id}] Chunk {chunk_id}: Rate limit (429). Retrying in {delay}s.")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"[{trace_id}] Chunk {chunk_id}: Final rate limit failure (429). Skipping.")
                        response.raise_for_status()  # Raise to be caught below

                response.raise_for_status()

                # Success: Write file
                with open(audio_path, 'wb') as f:
                    f.write(response.content)

                break  # Exit retry loop on success

            else:
                raise RuntimeError(
                    f"TTS service failed after {MAX_RETRIES} retries for chunk {chunk_id}")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"[{trace_id}] Failed chunk {chunk_id}: HTTP {e.response.status_code} - {e.response.text}")
            return False  # Failed, do not update manifest

        except Exception as e:
            logger.error(f"[{trace_id}] Failed chunk {chunk_id}: {e}")
            return False  # Failed, do not update manifest

        # --- Success: Update manifest (CRITICAL SECTION for Concurrency) ---
        # Reload manifest fresh to avoid stale read/write (CRITICAL for concurrency)
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            logger.error(
                f"[{trace_id}] Failed to reload manifest for chunk {chunk_id} post-generation. File written but manifest skipped.")
            return True  # Audio file exists, but manifest is inconsistent

        # Append new chunk data
        manifest['ready_chunks'].append({
            "chunk_id": chunk_id, "filename": audio_filename, "page": page,
            "text_snippet": chunk['text'][:50] + "...", "start_time": chunk['start_time'],
            "duration_seconds": chunk['duration_seconds'], "end_time": chunk['end_time'],
            "sentences": chunk.get('sentences', [])
        })
        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])

        # Write manifest using validation helper
        try:
            validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
            logger.info(f"[{trace_id}] Chunk {chunk_id} marked as ready in manifest.")
        except Exception:
            pass  # Validation wrapper logs error

        return True  # Chunk successfully processed and manifest updated

async def generate_audio_streaming(citation_json_path: Path, book_id: str, trace_id: str,
                                   manifest_path: Path, limit=None):
    """
    Stage 3: Generate audio chunks via TTS API using parallel execution (asyncio.gather).
    """
    if not citation_json_path.exists():
        logger.error(f"[{trace_id}] Stage 3 Error: Citation file not found: {citation_json_path}")
        return None

    with open(citation_json_path, 'r') as f:
        data = json.load(f)

    audio_dir = OUTPUT_DIR / book_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    chunks_to_process = data['chunks'][:limit] if limit else data['chunks']
    logger.info(
        f"[{trace_id}] Stage 3: Preparing {len(chunks_to_process)} chunks for parallel generation (Max {TTS_MAX_CONCURRENT_REQUESTS} concurrent requests)...")

    # --- NEW: Parallel Execution Core (CRITICAL) ---

    tasks = []
    for chunk in chunks_to_process:
        # Create a list of async tasks calling the single chunk handler
        task = generate_single_chunk(chunk, book_id, trace_id, manifest_path, logger)
        tasks.append(task)

    # Execute tasks concurrently, limited by the global semaphore
    try:
        # results is a list of booleans (True=success/skipped, False=failed)
        results = await asyncio.gather(*tasks)

    except Exception as e:
        # Catastrophic failure caught here (Disk I/O failure during helper, Container termination, etc.)
        logger.error(f"[{trace_id}] Stage 3 interrupted by catastrophic failure: {e}",
                     exc_info=True)

        # Write partial state (Task 3)
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            manifest['processing_status'] = 'stage_3_partial'
            manifest['error_message'] = f"Audio generation interrupted: {str(e)}"

            validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
            logger.warning(f"[{trace_id}] Wrote 'stage_3_partial' status. Job can be resumed.")

        except Exception as write_error:
            logger.error(f"[{trace_id}] Failed to write partial state: {write_error}")

        return None  # Exit pipeline function on interruption

    # Success case: Final return value used by run_full_pipeline for final status check
    logger.info(
        f"[{trace_id}] Parallel execution finished. {sum(results)}/{len(results)} chunks successful/skipped.")
    return audio_dir


# ========================================
# Stage 1: PDF Processing with Structural Index
# ========================================

def extract_text_blocks_with_coords(page):
    """
    Extract text blocks with bounding box coordinates from PDF page.
    Uses PyMuPDF's get_text("dict") for structured extraction.

    Args:
        page: fitz.Page object

    Returns:
        list: Text blocks with coordinates (span-level granularity)
    """
    blocks_data = []
    text_dict = page.get_text("dict")

    for block in text_dict["blocks"]:
        if block["type"] != 0:  # Skip non-text blocks
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                bbox = span["bbox"]  # [x0, y0, x1, y1]
                blocks_data.append({
                    "text": span["text"],
                    "bbox": {
                        "x": round(bbox[0], 2),
                        "y": round(bbox[1], 2),
                        "width": round(bbox[2] - bbox[0], 2),
                        "height": round(bbox[3] - bbox[1], 2)
                    },
                    "font_size": round(span["size"], 2),
                    "font_name": span["font"],
                    "block_type": "span"
                })

    return blocks_data


def build_page_sentences(text_blocks, coordinate_blocks):
    """
    Phase 1.1: Build structural index linking sentences to coordinate spans.

    Creates deterministic mapping by reconstructing full text, splitting into
    sentences, then finding character offsets and mapping to span indices.

    Args:
        text_blocks: List of text block strings (legacy format)
        coordinate_blocks: List of span dictionaries with coordinates

    Returns:
        list: Sentence objects with span_start_index and span_end_index
    """
    # Reconstruct full page text with character offsets for each span
    full_text = ""
    span_offsets = []  # (start_char, end_char, span_index)

    for span_idx, span in enumerate(coordinate_blocks):
        span_text = span.get("text", "")
        start_offset = len(full_text)
        full_text += span_text
        end_offset = len(full_text)
        span_offsets.append((start_offset, end_offset, span_idx))

    # Split into sentences using same logic as original Stage 2
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # Map each sentence to its span range
    page_sentences = []
    current_char_offset = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find character range of this sentence in full_text
        sentence_start = current_char_offset
        sentence_end = sentence_start + len(sentence)

        # Find all spans that overlap with this sentence
        overlapping_spans = []
        for start, end, span_idx in span_offsets:
            # Check if span overlaps with sentence range
            if not (end <= sentence_start or start >= sentence_end):
                overlapping_spans.append(span_idx)

        if overlapping_spans:
            page_sentences.append({
                "text": sentence,
                "span_start_index": overlapping_spans[0],
                "span_end_index": overlapping_spans[-1]
            })

        # Move offset past this sentence and any whitespace
        current_char_offset = sentence_end
        # Account for regex split whitespace
        while current_char_offset < len(full_text) and full_text[current_char_offset].isspace():
            current_char_offset += 1

    return page_sentences


def derive_book_id(title_or_filename: str) -> str:
    """
    Canonical book_id derivation - Single Source of Truth.
    """
    name = Path(title_or_filename).stem if Path(title_or_filename).suffix else title_or_filename
    sanitized = re.sub(r'[^\w\s-]', '', name)
    sanitized = sanitized.strip().replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized if sanitized else "unknown_document"


def atomic_write_manifest(manifest_path: Path, manifest_data: dict, logger):
    """
    Atomically write manifest using temp file + move pattern.
    Raises exception on failure to ensure pipeline halt (fail-fast policy).
    """
    temp_fd = None
    temp_path = None

    try:
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp"
        )
        temp_path = Path(temp_path_str)

        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        shutil.move(str(temp_path), str(manifest_path))
        logger.debug(f"Manifest written atomically: {manifest_path}")

    except Exception as e:
        logger.error(f"CRITICAL: Failed to write manifest at {manifest_path}: {e}", exc_info=True)

        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except:
                pass

        # CRITICAL: Re-raise exception to enforce pipeline halt
        raise e


def validate_and_write_manifest(manifest_path: Path, manifest_data: dict, trace_id: str, logger):
    """
    Validate manifest against schema and atomically write to disk.
    Enforces fail-fast policy for data integrity.
    """
    try:
        # 1. Validate the schema
        validated_manifest = ManifestSchema(**manifest_data)

        # 2. Write validated dict content atomically
        atomic_write_manifest(manifest_path, validated_manifest.dict(), logger)

        logger.debug(f"[{trace_id}] Manifest validated and written successfully: {manifest_path}")

    except ValidationError as e:
        logger.error(f"[{trace_id}] CRITICAL: Manifest validation failed for {manifest_path}: {e}")
        # Critical failure: re-raise to halt pipeline (fail-fast)
        raise e
    except Exception as e:
        # Atomic write failure (already logged in atomic_write_manifest)
        raise e

def process_pdf(pdf_filename: str):
    """
    Stage 1: Process PDF to structured JSON with coordinate index.

    Generates _raw.json containing:
    - metadata
    - content array with:
        - text_blocks (legacy compatibility)
        - coordinate_blocks (span-level coordinates)
        - page_sentences (NEW: structural index with span pointers)
    """
    pdf_path = INPUT_DIR / pdf_filename
    if not pdf_path.exists():
        logger.error(f"Error: File not found: {pdf_path}")
        return None

    cache_file_name = pdf_path.stem + "_raw.json"
    cache_file_path = CACHE_DIR / cache_file_name

    logger.info(f"Stage 1: Processing '{pdf_path.name}'...")
    output_data = {"metadata": {}, "content": []}

    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            output_data["metadata"] = {
                "title": meta.get("title", pdf_path.stem),
                "author": meta.get("author", "Unknown"),
                "source_filename": pdf_path.name,
                "total_pages": doc.page_count
            }

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)

                # Legacy text blocks (backward compatibility)
                blocks = page.get_text("blocks", sort=True)
                page_text_blocks = []
                for b in blocks:
                    block_text = b[4].replace("-\n", "").replace("\n", " ").strip()
                    if block_text:
                        page_text_blocks.append(block_text)

                # Coordinate blocks (span-level)
                coordinate_blocks = extract_text_blocks_with_coords(page)

                # Phase 1.1: Build structural index
                page_sentences = build_page_sentences(page_text_blocks, coordinate_blocks)

                if page_text_blocks or coordinate_blocks:
                    output_data["content"].append({
                        "page_number": page_num + 1,
                        "text_blocks": page_text_blocks,
                        "coordinate_blocks": coordinate_blocks,
                        "page_sentences": page_sentences  # NEW: Structural index
                    })

            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Stage 1 complete. Raw cache with structural index saved to: {cache_file_path}")
        return cache_file_path

    except Exception as e:
        logger.error(f"Error in Stage 1 processing PDF {pdf_path.name}: {e}", exc_info=True)
        return None


# ========================================
# Stage 2: Citation-Ready Chunks from Structural Index
# ========================================

def prepare_tts_chunks_with_citations(cache_file_path: Path, max_chars=400):
    """
    Stage 2: Prepare TTS chunks using pre-built structural index.

    Phase 1.3: Consumes page_sentences from _raw.json instead of manually splitting.
    Phase 1.4: Does NOT include content block in citation_ready.json output.
    Phase 1.5: Handles legacy _raw.json files gracefully.
    """
    if not cache_file_path or not cache_file_path.exists():
        logger.error(f"Stage 2 Error: Invalid cache file path: {cache_file_path}")
        return None

    logger.info(f"Starting Stage 2: Preparing citation-ready TTS chunks from structural index...")

    with open(cache_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tts_chunks = []
    global_sentence_index = 0
    estimated_total_time = 0.0
    AVG_CHARS_PER_SECOND = 14

    # Phase 1.5: Check for structural index
    has_structural_index = False
    if data['content'] and 'page_sentences' in data['content'][0]:
        has_structural_index = True
        logger.info("Structural index detected. Using indexed sentences.")
    else:
        logger.warning(
            "Legacy _raw.json detected (no structural index). Highlighting will be disabled.")

    try:
        for page_data in data['content']:
            page_num = page_data['page_number']

            # Phase 1.3: Use pre-built page_sentences if available
            if has_structural_index and 'page_sentences' in page_data:
                sentences_source = page_data['page_sentences']
            else:
                # Fallback: manual splitting for legacy files
                sentences_source = []
                for block_text in page_data.get('text_blocks', []):
                    block_sentences = re.split(r'(?<=[.!?])\s+', block_text)
                    for s in block_sentences:
                        if s.strip():
                            sentences_source.append({"text": s.strip()})

            # Chunk sentences based on character limit
            current_chunk = []
            current_chunk_chars = 0
            chunk_sentences_data = []

            for sentence_obj in sentences_source:
                sentence_text = sentence_obj['text'].strip()
                if not sentence_text:
                    continue

                sentence_chars = len(sentence_text) + 1

                if current_chunk_chars + sentence_chars < max_chars:
                    current_chunk.append(sentence_text)
                    current_chunk_chars += sentence_chars

                    # Build sentence data with structural indices if available
                    sentence_data = {
                        'global_index': global_sentence_index,
                        'sentence_in_chunk': len(chunk_sentences_data),
                        'text': sentence_text
                    }

                    # Phase 1.3: Include span indices if available
                    if has_structural_index and 'span_start_index' in sentence_obj:
                        sentence_data['span_start_index'] = sentence_obj['span_start_index']
                        sentence_data['span_end_index'] = sentence_obj['span_end_index']

                    chunk_sentences_data.append(sentence_data)
                    global_sentence_index += 1
                else:
                    # Flush current chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        est_duration = len(chunk_text) / AVG_CHARS_PER_SECOND
                        tts_chunks.append({
                            'chunk_id': len(tts_chunks),
                            'text': chunk_text,
                            'page': page_num,
                            'sentences': chunk_sentences_data,
                            'start_time': estimated_total_time,
                            'duration_seconds': est_duration,
                            'end_time': estimated_total_time + est_duration
                        })
                        estimated_total_time += est_duration

                    # Start new chunk
                    current_chunk = [sentence_text]
                    current_chunk_chars = sentence_chars

                    sentence_data = {
                        'global_index': global_sentence_index,
                        'sentence_in_chunk': 0,
                        'text': sentence_text
                    }

                    if has_structural_index and 'span_start_index' in sentence_obj:
                        sentence_data['span_start_index'] = sentence_obj['span_start_index']
                        sentence_data['span_end_index'] = sentence_obj['span_end_index']

                    chunk_sentences_data = [sentence_data]
                    global_sentence_index += 1

            # Flush final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                est_duration = len(chunk_text) / AVG_CHARS_PER_SECOND
                tts_chunks.append({
                    'chunk_id': len(tts_chunks),
                    'text': chunk_text,
                    'page': page_num,
                    'sentences': chunk_sentences_data,
                    'start_time': estimated_total_time,
                    'duration_seconds': est_duration,
                    'end_time': estimated_total_time + est_duration
                })
                estimated_total_time += est_duration

        book_id = derive_book_id(cache_file_path.stem)

        citation_file_name = book_id + '_citation_ready.json'
        citation_path = CACHE_DIR / citation_file_name

        # Phase 1.4: CRITICAL - Do NOT include content block
        output_data = {
            'metadata': data['metadata'],
            'book_id': book_id,
            'processing': {
                'total_chunks': len(tts_chunks),
                'total_sentences': global_sentence_index,
                'total_estimated_duration_seconds': estimated_total_time,
            },
            'chunks': tts_chunks,
            # Phase 1.5: Mark highlighting capability
            'highlighting_enabled': has_structural_index
        }

        # Original I/O replaced with atomic write
        # NOTE: No ManifestSchema validation here (intermediate file structure)
        try:
            atomic_write_manifest(citation_path, output_data, logger)
        except Exception as e:
            logger.error(f"Stage 2 failed during atomic write of citation file: {e}")
            return None

        logger.info(f"Stage 2 complete. Citation-ready chunks saved to: {citation_path}")
        logger.info(
            f"File size: {citation_path.stat().st_size / 1024:.1f} KB (coordinate data excluded)")

        return citation_path

    except Exception as e:
        logger.error(f"Error in Stage 2 preparing chunks: {e}", exc_info=True)
        return None


# ========================================
# Stage 3: Audio Generation
# ========================================

async def generate_audio_streaming(citation_json_path: Path, book_id: str, trace_id: str,
                                   manifest_path: Path, limit=None):
    """
    Stage 3: Generate audio chunks via TTS API with manifest tracking.
    Preserves all existing manifest fields (processing_status, trace_id, etc.)

    Args:
        citation_json_path: Path to citation_ready.json
        book_id: Canonical book identifier
        trace_id: Unique identifier for tracing
        manifest_path: Path to manifest.json
        limit: Optional limit on number of chunks to process
    """
    if not citation_json_path.exists():
        logger.error(f"[{trace_id}] Stage 3 Error: Citation file not found: {citation_json_path}")
        return None

    with open(citation_json_path, 'r') as f:
        data = json.load(f)

    # Ensure output directory exists
    audio_dir = OUTPUT_DIR / book_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Manifest Initialization/Loading ---
    manifest = {}
    if manifest_path.exists():
        try:
            # Load entire existing manifest to preserve all fields
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                logger.info(
                    f"[{trace_id}] Resuming job. Manifest state: {manifest.get('processing_status')}")
        except Exception as e:
            logger.error(
                f"[{trace_id}] CRITICAL: Failed to read existing manifest at {manifest_path}: {e}")
            logger.error(f"[{trace_id}] Halting audio generation to prevent data loss.")
            return None

    # Augment manifest fields if missing (from citation data)
    manifest.setdefault("metadata", data['metadata'])
    manifest.setdefault("book_id", book_id)
    manifest.setdefault("total_chunks", len(data['chunks']))
    manifest.setdefault("ready_chunks", [])
    manifest['processing_status'] = "stage_3_started"

    chunks_to_process = data['chunks'][:limit] if limit else data['chunks']
    logger.info(f"[{trace_id}] Stage 3: Generating audio for {len(chunks_to_process)} chunks...")

    # VALIDATION CALL #7: Write initial 'stage_3_started' status
    try:
        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
    except Exception:
        return None  # Pipeline halt

    # --- Audio Generation Loop ---
    for chunk in chunks_to_process:
        chunk_id = chunk['chunk_id']
        page = chunk['page']
        audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav"
        audio_path = audio_dir / audio_filename

        # Skip if already processed
        if audio_path.exists() or any(c['chunk_id'] == chunk_id for c in manifest['ready_chunks']):
            logger.debug(f"[{trace_id}] Skipping chunk {chunk_id} (already exists)")

            # VALIDATION CALL #8: Ensure manifest consistency
            if audio_path.exists() and not any(
                    c['chunk_id'] == chunk_id for c in manifest['ready_chunks']):
                logger.warning(f"[{trace_id}] Chunk {chunk_id} missing from manifest. Adding now.")

                # Load manifest fresh to prevent concurrency overwrite issues
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                except Exception:
                    logger.error(
                        f"[{trace_id}] Failed to reload manifest for chunk {chunk_id}. Skipping manifest update.")
                    continue

                manifest['ready_chunks'].append({
                    "chunk_id": chunk_id,
                    "filename": audio_filename,
                    "page": page,
                    "text_snippet": chunk['text'][:50] + "...",
                    "start_time": chunk['start_time'],
                    "duration_seconds": chunk['duration_seconds'],
                    "end_time": chunk['end_time'],
                    "sentences": chunk.get('sentences', [])
                })
                manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])

                try:
                    validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                except Exception:
                    continue

            continue

        # Generate audio via TTS API
        try:
            logger.info(
                f"[{trace_id}] Generating chunk {chunk_id + 1}/{len(chunks_to_process)} via API...")

            params = {
                "text": chunk['text'],
                "speaker_id": "",
                "style_wav": "",
                "language_id": "",
            }

            response = await client.post(TTS_SERVICE_URL, params=params, timeout=300.0)
            response.raise_for_status()

            with open(audio_path, 'wb') as f:
                f.write(response.content)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"[{trace_id}] Failed chunk {chunk_id}: HTTP {e.response.status_code} - {e.response.text}")
            continue
        except Exception as e:
            logger.error(f"[{trace_id}] Failed chunk {chunk_id}: {e}")
            continue

        # VALIDATION CALL #9: Update manifest with new chunk
        # Load manifest fresh to prevent concurrency overwrite issues
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            logger.error(
                f"[{trace_id}] Failed to reload manifest for chunk {chunk_id} before update. Skipping chunk addition.")
            continue

        manifest['ready_chunks'].append({
            "chunk_id": chunk_id,
            "filename": audio_filename,
            "page": page,
            "text_snippet": chunk['text'][:50] + "...",
            "start_time": chunk['start_time'],
            "duration_seconds": chunk['duration_seconds'],
            "end_time": chunk['end_time'],
            "sentences": chunk.get('sentences', [])
        })
        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])

        try:
            validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
            logger.info(f"[{trace_id}] Chunk {chunk_id} marked as ready in manifest.")
        except Exception:
            continue

    logger.info(f"[{trace_id}] Stage 3 complete. Audio saved to: {audio_dir}")
    return audio_dir


# ========================================
# Citation Lookup (API Helper)
# ========================================

# ~/TTS/my_app/pdf_processor/process.py

def get_citation_at_timestamp(citation_path: Path, timestamp_seconds: float):
    """
    Phase 2.3: Find citation and return structural indices only (minimal schema).
    FIX: Implemented epsilon tolerance to stabilize sentence indexing and prevent flickering.
    """
    EPSILON = 0.0001  # 100 microseconds of tolerance

    # Phase 2.1: Load data within helper scope
    try:
        with open(citation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load citation file {citation_path}: {e}")
        return None

    # Find matching chunk
    for chunk in data['chunks']:
        if chunk['start_time'] <= timestamp_seconds < chunk[
            'end_time'] + EPSILON:  # Add epsilon for safety
            time_into_chunk = timestamp_seconds - chunk['start_time']
            num_sentences = len(chunk['sentences'])

            if chunk['duration_seconds'] > 0 and num_sentences > 0:
                # Calculate ratio, add epsilon to ensure progress doesn't exceed 1.0 due to float errors
                progress_ratio = time_into_chunk / chunk['duration_seconds']

                # Clamp ratio to prevent indexing beyond the last sentence
                clamped_ratio = min(progress_ratio, 1.0 - EPSILON)

                sentence_index = int(clamped_ratio * num_sentences)

                # Final clamping, although min(ratio, 1.0 - EPSILON) should prevent maxing out
                sentence_index = max(0, min(sentence_index, num_sentences - 1))

            else:
                sentence_index = 0  # Default to first sentence

            sentence = chunk['sentences'][sentence_index]

            # Extract structural indices (defensive)
            span_start = sentence.get('span_start_index', -1)
            span_end = sentence.get('span_end_index', -1)

            # Phase 2.3: Return minimal schema
            return {
                'page': chunk['page'],
                'span_start_index': span_start,
                'span_end_index': span_end,
                # FIX: Access the top-level 'highlighting_enabled' flag
                'highlighting_enabled': data.get('highlighting_enabled', False)
            }

    logger.warning(f"No chunk found for timestamp: {timestamp_seconds}")
    return None