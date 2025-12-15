# ~/TTS/my_app/pdf_processor/process.py
"""
PDF Processing Service - V1.6
=============================
Orchestrates the PDF-to-Audiobook pipeline using extraction_engine.

Stages:
    1. Extraction (extraction_engine.extract_page)
    1.5. Normalization (extraction_engine.normalize_header_footer_across_document)
    2. Chunking (extraction_engine.compile_tts_ready_content)
    3. Audio Generation (TTS service)
"""

from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import sys
import json
from pathlib import Path
import logging
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import tempfile
import shutil
import datetime
import re
import unicodedata
import json

# V1.6: Use extraction_engine instead of text_cleanup
try:
    import pdf_processor.extraction_engine as extraction_engine
except ImportError:
    import extraction_engine


# ========================================
# Pydantic Schema Definition
# ========================================

class ReadyChunkSchema(BaseModel):
    chunk_id: int
    filename: str
    page: Optional[int] = None
    pages: List[int] = Field(default_factory=list)  # V1.5: Multi-page support
    text_snippet: str
    start_time: float
    duration_seconds: float
    end_time: float
    sentences: List[dict]


class ManifestSchema(BaseModel):
    # Core Metadata
    metadata: dict = Field(..., description="Original PDF metadata")
    book_id: str
    trace_id: str

    # State and Progress
    processing_status: Optional[str] = Field(None)
    total_chunks: int = Field(0)
    ready_chunks: List[ReadyChunkSchema] = Field([])

    # Error/Recovery
    error_message: Optional[str] = Field(None)


# ========================================
# Configuration & Setup
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PDFProcessorService")

BASE_DIR = Path("/workspace")
CACHE_DIR = BASE_DIR / "pdf_cache"
INPUT_DIR = BASE_DIR / "pdf_input"
OUTPUT_DIR = BASE_DIR / "outputs" / "audiobooks"

CACHE_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDF Processing Service")
client = httpx.AsyncClient(timeout=300.0)

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8000/api/tts")
TTS_MAX_CONCURRENT_REQUESTS = int(os.getenv('TTS_MAX_CONCURRENT_REQUESTS', '10'))
TTS_SEMAPHORE = asyncio.Semaphore(TTS_MAX_CONCURRENT_REQUESTS)
MANIFEST_LOCK = asyncio.Lock()

TTS_MAX_CHUNK_CHARS = 650

EXTRACTOR_VERSION = "2.1"

MAX_CONSECUTIVE_FAILURES = 5

# ========================================
# Lifecycle Events
# ========================================

@app.on_event("startup")
async def startup_event():
    TTS_SERVICE_ROOT = TTS_SERVICE_URL.replace("/api/tts", "/")
    if not TTS_SERVICE_URL or TTS_SERVICE_URL == "http://tts-service:8000/api/tts":
        logger.warning("TTS_SERVICE_URL not configured. Verify environment variable.")
    try:
        response = await client.get(TTS_SERVICE_ROOT)
        response.raise_for_status()
        logger.info(f"Successfully connected to TTS service at {TTS_SERVICE_ROOT}")
    except Exception as e:
        logger.error(f"Failed to connect to TTS service: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


# ========================================
# API Endpoints
# ========================================

@app.post("/api/v1/process/{pdf_filename}")
async def start_pdf_processing(pdf_filename: str, background_tasks: BackgroundTasks):
    safe_filename = re_sanitize(pdf_filename)
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    pdf_path = INPUT_DIR / safe_filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    book_id = derive_book_id(pdf_path.stem)
    trace_id = str(uuid.uuid4())
    audio_dir = OUTPUT_DIR / book_id
    manifest_path = audio_dir / "manifest.json"

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
        validate_and_write_manifest(manifest_path, stub_manifest, trace_id, logger)
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to create manifest stub: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize processing")

    background_tasks.add_task(run_full_pipeline, safe_filename, book_id, trace_id, False)

    return {"status": "processing_started", "book_id": book_id, "trace_id": trace_id}


@app.post("/api/v1/retry/{book_id}")
async def retry_processing(
        book_id: str,
        background_tasks: BackgroundTasks,
        force_rebuild: bool = False
):
    logger.info(f"--- RETRY DEBUG: Start for '{book_id}' ---")

    # 1. Resolve book ID
    safe_book_id = re_sanitize(book_id)
    manifest_path = OUTPUT_DIR / safe_book_id / "manifest.json"

    # 2. Try to find source filename from multiple sources
    source_filename = None
    existing_manifest = None

    # Source A: Existing manifest
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                existing_manifest = json.load(f)
            source_filename = existing_manifest.get('metadata', {}).get('source_filename')
            logger.info(f"Found source from manifest: {source_filename}")
        except Exception as e:
            logger.warning(f"Failed to read manifest: {e}")

    # Source B: Scan INPUT_DIR for matching PDF (Lead's approach)
    if not source_filename:
        for pdf_file in INPUT_DIR.glob("*.pdf"):
            if derive_book_id(pdf_file.stem) == safe_book_id:
                source_filename = pdf_file.name
                logger.info(f"Found source from INPUT_DIR scan: {source_filename}")
                break

    # Source C: Try common naming patterns (Specialist addition)
    if not source_filename:
        candidates = [
            f"{safe_book_id}.pdf",
            f"{safe_book_id.replace('_', ' ')}.pdf",
            f"{safe_book_id.replace('_', '-')}.pdf",
        ]
        for candidate in candidates:
            if (INPUT_DIR / candidate).exists():
                source_filename = candidate
                logger.info(f"Found source from pattern match: {source_filename}")
                break

    # 3. Final check - must have source file (Specialist addition)
    if not source_filename:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for book_id '{safe_book_id}'. Check INPUT_DIR."
        )

    # Verify the PDF actually exists (Specialist addition)
    if not (INPUT_DIR / source_filename).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source PDF '{source_filename}' not found in INPUT_DIR."
        )

    # 4. Status check (only if not force_rebuild and manifest exists)
    if not force_rebuild:
        if not existing_manifest:
            raise HTTPException(
                status_code=404,
                detail=f"No manifest found for '{safe_book_id}'. Use force_rebuild=true."
            )
        if existing_manifest.get('processing_status') not in ['failed', 'stage_3_partial']:
            raise HTTPException(
                status_code=400,
                detail="Job is not in a retryable state. Use force_rebuild=true."
            )

    new_trace_id = str(uuid.uuid4())

    # 5. SCORCHED EARTH (Combined approach)
    if force_rebuild:
        logger.warning(f"Force Rebuild: Scorched Earth for '{safe_book_id}'")

        # A. Nuke output directory
        book_dir = OUTPUT_DIR / safe_book_id
        if book_dir.exists():
            shutil.rmtree(book_dir)
            logger.info(f"Deleted output directory: {book_dir}")
        book_dir.mkdir(parents=True, exist_ok=True)

        # B. Nuke caches - Glob pattern approach (Lead's approach)
        pdf_stem = Path(source_filename).stem

        # Use set to avoid duplicate deletion attempts
        files_to_nuke = set()
        files_to_nuke.update(CACHE_DIR.glob(f"*{safe_book_id}*"))
        files_to_nuke.update(CACHE_DIR.glob(f"*{pdf_stem}*"))

        # Delete with error handling (Lead's approach)
        for file_path in files_to_nuke:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    logger.info(f"Deleted cache: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path.name}: {e}")

        # C. Fresh manifest
        manifest = {
            "metadata": {"source_filename": source_filename},
            "book_id": safe_book_id,
            "trace_id": new_trace_id,
            "processing_status": "processing_started",
            "total_chunks": 0,
            "ready_chunks": [],
            "error_message": None
        }
    else:
        manifest = existing_manifest
        manifest['processing_status'] = 'processing_started'
        manifest['trace_id'] = new_trace_id
        manifest['error_message'] = None

    # 6. Write manifest and start pipeline
    validate_and_write_manifest(manifest_path, manifest, new_trace_id, logger)

    background_tasks.add_task(
        run_full_pipeline, source_filename, safe_book_id, new_trace_id, force_rebuild
    )

    return {"status": "retry_started", "book_id": safe_book_id, "trace_id": new_trace_id}


@app.get("/api/v1/citation/{book_id}")
async def get_citation(book_id: str, timestamp: float = 0.0):
    safe_book_id = re_sanitize(book_id)
    safe_book_id_sanitized = safe_book_id.replace(' ', '_')
    citation_path = CACHE_DIR / f"{safe_book_id_sanitized}_citation_ready.json"

    if not citation_path.exists():
        for f in CACHE_DIR.glob(f"*{safe_book_id_sanitized}*citation_ready.json"):
            citation_path = f
            break
        else:
            raise HTTPException(status_code=404, detail="Citation not found")

    citation_data = get_citation_at_timestamp(citation_path, timestamp)
    if not citation_data:
        raise HTTPException(status_code=404, detail="Timestamp out of range")
    return citation_data


@app.get("/api/v1/document/{pdf_filename}")
async def serve_pdf_document(pdf_filename: str):
    safe_filename = re_sanitize(pdf_filename)
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400)

    file_path = INPUT_DIR / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404)

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{safe_filename}\""}
    )


# ========================================
# Pipeline Orchestrator
# ========================================

async def run_full_pipeline(
        pdf_filename: str,
        book_id: str,
        trace_id: str,
        force_rebuild: bool = False
):
    logger.info(f"[{trace_id}] Pipeline started for: {pdf_filename}")

    pdf_path = INPUT_DIR / pdf_filename
    raw_cache_path = CACHE_DIR / f"{pdf_path.stem}_raw.json"
    citation_filename = f"{book_id}_citation_ready.json"
    citation_path = CACHE_DIR / citation_filename
    manifest_path = OUTPUT_DIR / book_id / "manifest.json"

    try:
        # ====================================================================
        # P0 FIX: Version-Aware Cache Check
        # ====================================================================
        use_cache = False

        if not force_rebuild and citation_path.exists():
            # Check 1: Timestamp freshness
            cache_fresh = not (pdf_path.stat().st_mtime > citation_path.stat().st_mtime)

            # Check 2: Extractor version match
            version_match = False
            try:
                with open(citation_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                cached_version = cached_data.get('metadata', {}).get('extractor_version')
                version_match = (cached_version == EXTRACTOR_VERSION)

                if not version_match:
                    logger.info(
                        f"[{trace_id}] Cache version mismatch: "
                        f"cached={cached_version}, current={EXTRACTOR_VERSION}. Rebuilding."
                    )
            except Exception as e:
                logger.warning(f"[{trace_id}] Failed to read cache for version check: {e}")

            use_cache = cache_fresh and version_match

        if use_cache:
            logger.info(f"[{trace_id}] Cache hit (v{EXTRACTOR_VERSION}). Skipping extraction.")
            with open(citation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            m['metadata'] = data['metadata']
            m['total_chunks'] = data['processing']['total_chunks']
            m['processing_status'] = 'stage_2_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        else:
            # Stage 1: Extraction (extraction_engine)
            logger.info(
                f"[{trace_id}] Stage 1: Extraction (extraction_engine v{EXTRACTOR_VERSION})...")
            raw_cache_path = process_pdf(pdf_filename, book_id, trace_id)
            if not raw_cache_path:
                raise RuntimeError("Stage 1 failed")

            # Update Manifest
            with open(manifest_path, 'r') as f:
                m = json.load(f)
            m['processing_status'] = 'stage_1_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)

            # Stage 2: Semantic Chunking
            logger.info(f"[{trace_id}] Stage 2: Semantic Chunking...")
            citation_path = prepare_tts_chunks_with_citations(raw_cache_path, trace_id)
            if not citation_path:
                raise RuntimeError("Stage 2 failed")

            # Update Manifest
            with open(citation_path, 'r') as f:
                data = json.load(f)
            with open(manifest_path, 'r') as f:
                m = json.load(f)
            m['metadata'] = data['metadata']
            m['total_chunks'] = data['processing']['total_chunks']
            m['processing_status'] = 'stage_2_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)

        # Stage 3: Audio Generation
        logger.info(f"[{trace_id}] Stage 3: Audio Generation...")
        with open(manifest_path, 'r') as f:
            m = json.load(f)
        m['processing_status'] = 'stage_3_started'
        validate_and_write_manifest(manifest_path, m, trace_id, logger)

        await generate_audio_streaming(citation_path, book_id, trace_id, manifest_path)

        # Final Status Check with Reconciliation
        final_status = reconcile_manifest_with_disk(book_id, manifest_path, trace_id)
        logger.info(f"[{trace_id}] Job Finished: {final_status}")

    except Exception as e:
        logger.error(f"[{trace_id}] Critical Failure: {e}", exc_info=True)
        try:
            with open(manifest_path, 'r') as f:
                m = json.load(f)
            m['processing_status'] = 'failed'
            m['error_message'] = str(e)
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        except:
            pass


# ========================================
# STAGE 1: Extraction (extraction_engine)
# ========================================

def process_pdf(pdf_filename: str, book_id: str = None, trace_id: str = None):
    """
    STAGE 1: PDF Extraction using extraction_engine.

    V1.6: Delegates all extraction logic to the canonical extraction engine.
    """
    pdf_path = INPUT_DIR / pdf_filename
    if not pdf_path.exists():
        return None

    cache_file_path = CACHE_DIR / f"{pdf_path.stem}_raw.json"
    logger.info(f"[{trace_id or 'N/A'}] Stage 1: Extraction started for {pdf_path.name}")

    try:
        with fitz.open(pdf_path) as doc:
            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", pdf_path.stem),
                "author": doc.metadata.get("author", "Unknown"),
                "source_filename": pdf_path.name,
                "total_pages": doc.page_count,
                "extractor_version": EXTRACTOR_VERSION  # NEW
            }

            # Stage 1: Extract all pages using extraction_engine
            page_outputs = []
            for page_num in range(doc.page_count):
                page_data = extraction_engine.extract_page(doc, page_num, trace_id)
                page_outputs.append(page_data)

            # Stage 1.5: Normalize headers/footers across document
            extraction_engine.normalize_header_footer_across_document(page_outputs, trace_id=trace_id)

        # Build output structure compatible with Stage 2
        output_data = {
            "metadata": metadata,
            "pages": page_outputs,
            "book_id": book_id or derive_book_id(pdf_path.stem),
            "trace_id": trace_id
        }

        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(
            f"[{trace_id or 'N/A'}] Stage 1 complete: {len(page_outputs)} pages extracted"
        )
        return cache_file_path

    except Exception as e:
        logger.error(f"[{trace_id or 'N/A'}] Stage 1 failed: {e}", exc_info=True)
        return None


# ========================================
# STAGE 2: Semantic Chunking (extraction_engine)
# ========================================

def prepare_tts_chunks_with_citations(cache_file_path: Path, trace_id: str = None):
    """
    STAGE 2: Semantic Chunking using extraction_engine.

    V1.6: Delegates all chunking logic to the canonical extraction engine.
    """
    if not cache_file_path.exists():
        return None

    with open(cache_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"[{trace_id or 'N/A'}] Stage 2: Semantic chunking started")

    try:
        # Get page outputs from Stage 1
        page_outputs = data.get('pages', [])

        if not page_outputs:
            logger.error(f"[{trace_id or 'N/A'}] Stage 2: No pages found in cache")
            return None

        # Stage 2: Compile TTS-ready content using extraction_engine
        tts_result = extraction_engine.compile_tts_ready_content(page_outputs, trace_id)

        # Build citation-ready output
        book_id = data.get('book_id') or derive_book_id(cache_file_path.stem)
        citation_path = CACHE_DIR / f"{book_id}_citation_ready.json"

        output_data = {
            'metadata': data['metadata'],
            'book_id': book_id,
            'trace_id': trace_id,
            'processing': tts_result['processing'],
            'document_headers': tts_result.get('document_headers', []),
            'document_footers': tts_result.get('document_footers', []),
            'chunks': _format_chunks_for_manifest(tts_result['chunks']),
            'highlighting_enabled': True
        }

        atomic_write_manifest(citation_path, output_data, logger)
        logger.info(
            f"[{trace_id or 'N/A'}] Stage 2 complete: "
            f"{tts_result['processing']['total_chunks']} chunks, "
            f"{tts_result['processing']['total_sentences']} sentences"
        )
        return citation_path

    except Exception as e:
        logger.error(f"[{trace_id or 'N/A'}] Stage 2 failed: {e}", exc_info=True)
        return None


def _format_chunks_for_manifest(chunks: List[dict]) -> List[dict]:
    """
    Formats extraction_engine chunks for manifest/citation compatibility.
    """
    formatted = []
    for chunk in chunks:
        formatted.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'page': chunk['page'],
            'pages': chunk.get('pages', [chunk['page']]),
            'sentences': _format_sentences_for_manifest(chunk['sentences']),
            'start_time': chunk['start_time'],
            'duration_seconds': chunk['duration_seconds'],
            'end_time': chunk['end_time']
        })
    return formatted


def _format_sentences_for_manifest(sentences: List[dict]) -> List[dict]:
    """
    Formats extraction_engine sentences for manifest compatibility.

    V1.9 Phase 1: Now includes per-sentence timing fields.
    """
    formatted = []
    for i, sent in enumerate(sentences):
        formatted.append({
            'global_index': sent.get('global_index', i),
            'sentence_in_chunk': i,
            'text': sent.get('tts_text', sent.get('text', '')),
            'span_start_index': sent.get('span_start_index', 0),
            'span_end_index': sent.get('span_end_index', 0),
            'page_number': sent.get('page_number'),
            'role': sent.get('role', 'body'),
            'is_stitched': sent.get('is_stitched', False),
            # V1.9: Per-sentence timing
            'start_time': sent.get('start_time', 0.0),
            'end_time': sent.get('end_time', 0.0),
            'duration_seconds': sent.get('duration_seconds', 0.0),
        })
    return formatted


# ========================================
# STAGE 3: TTS Generation (Unchanged)
# ========================================

async def generate_single_chunk(
        chunk: dict,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        logger
):
    chunk_id = chunk['chunk_id']
    page = chunk['page']
    audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav"
    audio_path = OUTPUT_DIR / book_id / audio_filename

    async with TTS_SEMAPHORE:
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except:
            return False

        already_in_manifest = any(c['chunk_id'] == chunk_id for c in manifest['ready_chunks'])

        if audio_path.exists():
            if not already_in_manifest:
                async with MANIFEST_LOCK:
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        manifest['ready_chunks'].append({
                            "chunk_id": chunk_id,
                            "filename": audio_filename,
                            "page": page,
                            "pages": chunk.get('pages') or ([page] if page is not None else []),
                            "text_snippet": chunk['text'][:50] + "...",
                            "start_time": chunk['start_time'],
                            "duration_seconds": chunk['duration_seconds'],
                            "end_time": chunk['end_time'],
                            "sentences": chunk.get('sentences', [])
                        })
                        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])
                        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                        logger.info(f"[{trace_id}] Chunk {chunk_id} recovered from disk.")
                    except Exception as e:
                        logger.error(f"[{trace_id}] Failed to recover chunk {chunk_id}: {e}",
                                     exc_info=True)
            return True

        if already_in_manifest:
            return True

        # ====================================================================
        # P0 FIX: TTS Overflow Guard (Fail Fast)
        # ====================================================================
        text_to_send = chunk['text']
        if len(text_to_send) > TTS_MAX_CHUNK_CHARS:
            logger.error(
                f"[{trace_id}] Chunk {chunk_id} REJECTED: {len(text_to_send)} chars "
                f"exceeds TTS safety limit ({TTS_MAX_CHUNK_CHARS}). "
                f"Text preview: '{text_to_send[:80]}...'"
            )
            return False  # Fail fast — don't hang TTS service for 60s

        # Generate new audio
        try:
            params = {
                "text": text_to_send,
                "speaker_id": "",
                "style_wav": "",
                "language_id": ""
            }
            response = await client.post(TTS_SERVICE_URL, data=params, timeout=300.0)
            response.raise_for_status()

            with open(audio_path, 'wb') as f:
                f.write(response.content)

        except Exception as e:
            logger.error(f"[{trace_id}] TTS Failed Chunk {chunk_id}: {e}")
            return False

        # Atomic Manifest Update
        async with MANIFEST_LOCK:
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                manifest['ready_chunks'].append({
                    "chunk_id": chunk_id,
                    "filename": audio_filename,
                    "page": page,
                    "pages": chunk.get('pages') or ([page] if page is not None else []),
                    "text_snippet": chunk['text'][:50] + "...",
                    "start_time": chunk['start_time'],
                    "duration_seconds": chunk['duration_seconds'],
                    "end_time": chunk['end_time'],
                    "sentences": chunk.get('sentences', [])
                })
                manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])
                validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                logger.info(f"[{trace_id}] Chunk {chunk_id} ready.")
            except Exception as e:
                logger.error(f"[{trace_id}] Failed to update manifest for chunk {chunk_id}: {e}",
                             exc_info=True)

        return True


async def generate_audio_streaming(
        citation_json_path: Path,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        limit=None
):
    if not citation_json_path.exists():
        return None

    with open(citation_json_path, 'r') as f:
        data = json.load(f)

    (OUTPUT_DIR / book_id).mkdir(parents=True, exist_ok=True)

    chunks = data['chunks'][:limit] if limit else data['chunks']
    tasks = [
        generate_single_chunk(c, book_id, trace_id, manifest_path, logger)
        for c in chunks
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if r is True)
        failures = sum(1 for r in results if r is not True)
        logger.info(f"[{trace_id}] Stage 3 Summary: {successes}/{len(chunks)} chunks generated")
        if failures > 0:
            logger.warning(f"[{trace_id}] {failures} chunks failed to generate")
        return successes == len(chunks)
    except Exception as e:
        logger.error(f"[{trace_id}] Stage 3 Crash: {e}", exc_info=True)
        try:
            with open(manifest_path, 'r') as f:
                m = json.load(f)
            m['processing_status'] = 'stage_3_partial'
            m['error_message'] = f"Stage 3 crashed: {str(e)}"
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        except Exception as manifest_err:
            logger.error(f"[{trace_id}] Failed to update manifest after crash: {manifest_err}")
        return False


# ========================================
# Utilities
# ========================================

def reconcile_manifest_with_disk(book_id: str, manifest_path: Path, trace_id: str):
    """
    Final pass: Ensure manifest ready_chunks matches actual files on disk.
    """
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        audio_dir = OUTPUT_DIR / book_id
        actual_files = {f.name for f in audio_dir.glob("chunk_*.wav")}
        manifest_files = {c['filename'] for c in manifest['ready_chunks']}

        # Remove entries for missing files
        orphaned = manifest_files - actual_files
        if orphaned:
            manifest['ready_chunks'] = [
                c for c in manifest['ready_chunks'] if c['filename'] not in orphaned
            ]
            logger.warning(f"[{trace_id}] Removed {len(orphaned)} orphaned manifest entries")

        # Log any files missing from manifest
        missing = actual_files - manifest_files
        if missing:
            logger.warning(f"[{trace_id}] {len(missing)} audio files not in manifest: {missing}")

        # Update status based on reconciled count
        completed = len(manifest['ready_chunks'])
        total = manifest['total_chunks']

        if completed == total:
            manifest['processing_status'] = 'stage_3_complete'
        elif completed > 0:
            manifest['processing_status'] = 'stage_3_partial'
        else:
            manifest['processing_status'] = 'stage_3_failed'

        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
        return manifest['processing_status']

    except Exception as e:
        logger.error(f"[{trace_id}] Reconciliation failed: {e}")
        return None


def re_sanitize(filename):
    return re.sub(r'[^\w\s\-\.]', '', filename).strip()


def derive_book_id(title):
    return re.sub(r'_+', '_', re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_'))


def atomic_write_manifest(path, data, logger):
    tmp = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_path)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(str(tmp), str(path))
    except Exception as e:
        if tmp and tmp.exists():
            tmp.unlink()
        raise e


def validate_and_write_manifest(path, data, trace_id, logger):
    try:
        validated = ManifestSchema(**data)
        atomic_write_manifest(path, validated.dict(), logger)
    except Exception as e:
        logger.error(f"Manifest Error: {e}")
        raise e


def get_citation_at_timestamp(path, timestamp):
    """
    API implementation for citation lookup.

    V1.9 Phase 1 Fix #4: Uses per-sentence timing instead of interpolation.
    Falls back to interpolation for legacy manifests without sentence timing.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)

        for chunk in data['chunks']:
            # Check if timestamp falls within this chunk
            if chunk['start_time'] <= timestamp < chunk['end_time'] + 0.0001:
                sentences = chunk.get('sentences', [])

                if not sentences:
                    continue

                # V1.9: Try per-sentence timing first
                for s in sentences:
                    sent_start = s.get('start_time')
                    sent_end = s.get('end_time')

                    # Use per-sentence timing if available
                    if sent_start is not None and sent_end is not None:
                        if sent_start <= timestamp < sent_end + 0.0001:
                            return {
                                'page': s.get('page_number', chunk['page']),
                                'span_start_index': s['span_start_index'],
                                'span_end_index': s['span_end_index'],
                                'role': s.get('role', 'body'),
                                'highlighting_enabled': data.get('highlighting_enabled', True),
                                'sentence_text': s.get('text', '')[:50],
                            }

                # Fallback: Legacy interpolation for old manifests
                prog = (timestamp - chunk['start_time']) / chunk['duration_seconds']
                idx = int(min(prog, 0.999) * len(sentences))
                s = sentences[idx]

                return {
                    'page': s.get('page_number', chunk['page']),
                    'span_start_index': s['span_start_index'],
                    'span_end_index': s['span_end_index'],
                    'role': s.get('role', 'body'),
                    'highlighting_enabled': data.get('highlighting_enabled', True),
                    'sentence_text': s.get('text', '')[:50],
                }
    except Exception:
        pass

    return None