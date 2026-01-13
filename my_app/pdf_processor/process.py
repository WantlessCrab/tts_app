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

EXTRACTOR_VERSION = "2.2"

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

            # ==========================================================
            # Phase-A Pre-Scan: Global Document Metrics (HARDENED)
            # ==========================================================
            global_line_gaps = []
            global_font_sizes = []

            # PERFORMANCE HARDENING:
            # Sample a bounded set of pages to avoid double-parsing entire books.
            total_pages = doc.page_count
            sample_pages = set()

            # First pages (skip title page noise by starting at page 1 if possible)
            start_page = 1 if total_pages > 2 else 0
            for i in range(start_page, min(20, total_pages)):
                sample_pages.add(i)

            # Middle slice (captures body text)
            if total_pages > 40:
                mid_start = total_pages // 2 - 5
                for i in range(mid_start, min(mid_start + 10, total_pages)):
                    sample_pages.add(i)

            # Final pages (indexes / appendices)
            for i in range(max(0, total_pages - 10), total_pages):
                sample_pages.add(i)

            for page_num in sorted(sample_pages):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")

                for block in text_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    lines = block.get("lines", [])

                    # -----------------------------
                    # FONT SIZE SAMPLING (Span-Level)
                    # -----------------------------
                    for line in lines:
                        for span in line.get("spans", []):
                            size = span.get("size")
                            if size:
                                global_font_sizes.append(float(size))

                    # ----------------------------------
                    # LINE HEIGHT / LEADING (Line-to-Line)
                    # ----------------------------------
                    for i in range(1, len(lines)):
                        prev_bbox = lines[i - 1].get("bbox")
                        curr_bbox = lines[i].get("bbox")

                        if not prev_bbox or not curr_bbox:
                            continue

                        prev_bottom = prev_bbox[3]
                        curr_top = curr_bbox[1]
                        gap = curr_top - prev_bottom

                        # Filter implausible values (noise / layout jumps)
                        if 0 < gap < 200:
                            global_line_gaps.append(gap)

            GLOBAL_MEDIAN_LINE_HEIGHT = (
                sorted(global_line_gaps)[len(global_line_gaps) // 2]
                if global_line_gaps else None
            )

            GLOBAL_MEDIAN_FONT_SIZE = (
                sorted(global_font_sizes)[len(global_font_sizes) // 2]
                if global_font_sizes else None
            )

            logger.info(
                f"[{trace_id}] Global Metrics: "
                f"MedianFont={GLOBAL_MEDIAN_FONT_SIZE if GLOBAL_MEDIAN_FONT_SIZE is not None else 'N/A'}, "
                f"MedianLineGap={GLOBAL_MEDIAN_LINE_HEIGHT if GLOBAL_MEDIAN_LINE_HEIGHT is not None else 'N/A'} "
                f"(Sampled {len(sample_pages)} pages)"
            )
            # ------------------------------------

            # Extract metadata
            metadata = {
                "global_median_line_height": GLOBAL_MEDIAN_LINE_HEIGHT,
                "global_median_font_size": GLOBAL_MEDIAN_FONT_SIZE,
                "title": doc.metadata.get("title", pdf_path.stem),
                "author": doc.metadata.get("author", "Unknown"),
                "source_filename": pdf_path.name,
                "total_pages": doc.page_count,
                "extractor_version": EXTRACTOR_VERSION  # NEW
            }

            # Stage 1: Extract all pages using extraction_engine
            # IMPORTANT:
            # extract_page() returns spans in authoritative visual reading order.
            # Downstream code MUST NOT re-sort spans or override paragraph_index.
            page_outputs = []
            for page_num in range(doc.page_count):
                page_data = extraction_engine.extract_page(
                    doc,
                    page_num,
                    trace_id=trace_id,
                    global_median_line_height=GLOBAL_MEDIAN_LINE_HEIGHT,
                    global_median_font_size=GLOBAL_MEDIAN_FONT_SIZE
                )
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

        # ═══════════════════════════════════════════════════════════════════
        # SEMANTIC ARTIFACT: Persist RONC/A2/disposition decisions
        # P6 FIX: Use processed_spans from tts_result (contains P6 modifications)
        # instead of page_outputs (original unmodified spans)
        # ═══════════════════════════════════════════════════════════════════
        semantic_path = CACHE_DIR / f"{book_id}_semantic.json"
        processed_spans = tts_result.get('processed_spans', {})
        _save_semantic_artifact(
            processed_spans,
            semantic_path,
            book_id,
            trace_id,
            data['metadata']
        )
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

    V2.1: Includes full provenance for joinability and forensic debugging.

    ARCHITECTURAL CONTRACT:
    - Manifest sentences MUST be joinable back to source spans
    - Provenance fields enable late-stage QA and error triage
    - Runtime-only fields (_source_spans) are stripped; IDs are preserved
    - Flow identity is PROJECTED from spans here, not stored on sentences

    Provenance fields:
    - source_cids: Canonical span IDs (reversible join key)
    - source_unit_ids: RONC atomic unit memberships
    - source_flow_ids: Layout stream identities (derived from spans)
    - is_multi_flow: Contamination flag (sentence spans multiple streams)
    - is_stitched: Whether sentence was merged from fragments
    """
    formatted = []
    for i, sent in enumerate(sentences):
        # ═══════════════════════════════════════════════════════════════════
        # Core fields (existing)
        # ═══════════════════════════════════════════════════════════════════
        entry = {
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
            'paragraph_index': sent.get('paragraph_index'),
        }

        # ═══════════════════════════════════════════════════════════════════
        # RONC v2.1: Provenance fields (JOIN KEYS — non-negotiable)
        # ═══════════════════════════════════════════════════════════════════

        # Source span canonical IDs (primary join key)
        source_cids = sent.get('_source_span_ids')
        if source_cids:
            # Filter None sentinels for manifest (keep only valid IDs)
            entry['source_cids'] = [cid for cid in source_cids if cid is not None]

        # RONC atomic unit memberships
        source_units = sent.get('_ronc_atomic_units')
        if source_units:
            entry['source_unit_ids'] = source_units

        # ═══════════════════════════════════════════════════════════════════
        # RONC v2.1: Flow identity PROJECTION (derived from spans at emission)
        # Flow identity is NOT stored on sentences — it is projected fresh
        # from authoritative span sources to maintain RONC as single truth.
        # ═══════════════════════════════════════════════════════════════════
        source_spans = sent.get('_source_spans') or []
        flow_ids = set()
        for sp in source_spans:
            flow = sp.get('layout_stream')
            if flow:
                flow_ids.add(flow)

        if flow_ids:
            entry['source_flow_ids'] = sorted(flow_ids)
            if len(flow_ids) > 1:
                entry['is_multi_flow'] = True

        # ═══════════════════════════════════════════════════════════════════
        # Audit flags (cross-boundary merges detected during stitching)
        # ═══════════════════════════════════════════════════════════════════
        if sent.get('_ronc_cross_unit_merge'):
            entry['is_cross_unit_merge'] = True

        if sent.get('_ronc_cross_flow_merge'):
            entry['is_cross_flow_merge'] = True

        # ═══════════════════════════════════════════════════════════════════
        # Diagnostic fields (optional but valuable for triage)
        # ═══════════════════════════════════════════════════════════════════
        if sent.get('_contaminated'):
            entry['is_contaminated'] = True
            entry['contaminated_roles'] = sent.get('_contaminated_roles', [])

        if sent.get('alignment_method'):
            entry['alignment_method'] = sent['alignment_method']

        if sent.get('boundary_risks'):
            entry['boundary_risks'] = sent['boundary_risks']

        formatted.append(entry)

    return formatted


# ========================================
# SEMANTIC ARTIFACT (Stage 2 Output Layer)
# ========================================

def _save_semantic_artifact(
        processed_spans: dict,
        output_path: Path,
        book_id: str,
        trace_id: str,
        metadata: dict
):
    """
    STAGE 2 OUTPUT: Semantic authority artifact.

    P6 FIX: Now receives processed_spans dict (keyed by CID) directly from
    compile_tts_ready_content, which includes P6 same-line promotions.

    Schema v1.0 Contract:
        - extraction artifact = geometry + basic classification (Stage 1)
        - semantic artifact = RONC authority + dispositions (Stage 2)
        - manifest = final TTS output (Stage 3)

    Scope clarification:
        - _tts_excluded is a Stage 2 eligibility decision
        - manifest.json is the emission authority (Stage 3)
    """
    semantic_data = {
        "metadata": {
            **metadata,
            "artifact_type": "semantic",
            "schema_version": "1.1",
            "stage": "2",
            "authority_scope": "semantic_eligibility",
        },
        "book_id": book_id,
        "trace_id": trace_id,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "spans": {}
    }

    # P6 FIX: Iterate processed_spans dict directly (already keyed by CID)
    for cid, sp in processed_spans.items():
        if not cid:
            continue

        semantic_data["spans"][cid] = {
                # RONC Contract (full authority record)
                "_ronc_contract": sp.get("_ronc_contract"),
                # Canonical ID (P5 FIX: required for provenance tracking)
                "_canonical_span_id": cid,
                # A2 Edge Qualification (Phase 7)
                "_a2_edge_exists": sp.get("_a2_edge_exists"),
                "_a2_cross_stream": sp.get("_a2_cross_stream"),
                "_a2_qualified": sp.get("_a2_qualified"),
                "_a2_edge_prev_id": sp.get("_a2_edge_prev_id"),
                # Semantic Disposition (with confidence)
                "_semantic_disposition": sp.get("_semantic_disposition"),
                "_semantic_reasons": sp.get("_semantic_reasons"),
                "_semantic_confidence": sp.get("_semantic_confidence"),
                # TTS Eligibility (Stage 2 decision, not emission truth)
                "_tts_excluded": sp.get("_tts_excluded"),
                "_tts_exclude_reason": sp.get("_tts_exclude_reason"),
                "_tts_include_reason": sp.get("_tts_include_reason"),
                # RONC Legacy Fields
                "_ronc_atomic_unit_id": sp.get("_ronc_atomic_unit_id"),
                "_ronc_atomic_role": sp.get("_ronc_atomic_role"),
                "_ronc_break_after": sp.get("_ronc_break_after"),
                "_ronc_rescue_applied": sp.get("_ronc_rescue_applied"),
                # Structural Context (for diagnostics)
                "layout_stream": sp.get("layout_stream"),
                "role": sp.get("role"),
                "page_number": sp.get("page_number"),
                "block_id": sp.get("block_id"),
                "cleaned_text": (sp.get("cleaned_text") or "")[:100],
                "line_index": sp.get("line_index"),
                "span_index_in_line": sp.get("span_index_in_line"),
                "line_id": sp.get("line_id"),
                "bbox": sp.get("bbox"),

                # ─────────────────────────────────────────────────────
                # Phase 1.3 Line-Aware Rescue Audit
                # ─────────────────────────────────────────────────────
                "_tts_rescued": sp.get("_tts_rescued"),
                "_tts_rescue_reason": sp.get("_tts_rescue_reason"),
                "_tts_promoted_to_body_stream": sp.get("_tts_promoted_to_body_stream"),
                "_tts_promotion_reason": sp.get("_tts_promotion_reason"),
                "_tts_inline_detection_method": sp.get("_tts_inline_detection_method"),

                # ─────────────────────────────────────────────────────
                # Phase 1.5 Continuity Override Audit
                # ─────────────────────────────────────────────────────
                "_continuity_override": sp.get("_continuity_override"),
                "_continuity_override_reason": sp.get("_continuity_override_reason"),
                "_original_geometry_role": sp.get("_original_geometry_role"),

                # ─────────────────────────────────────────────────────
                # P6 Same-Line Promotion Audit
                # ─────────────────────────────────────────────────────
                "_same_line_promoted": sp.get("_same_line_promoted"),
                "_zombie_role_fixed": sp.get("_zombie_role_fixed"),
                "_original_role": sp.get("_original_role"),
                "_original_layout_stream": sp.get("_original_layout_stream"),
            }

    semantic_data["summary"] = _build_semantic_summary(semantic_data["spans"])
    atomic_write_manifest(output_path, semantic_data, logger)
    logger.info(
        f"[{trace_id}] Semantic artifact saved: {len(semantic_data['spans'])} spans"
    )


def _build_semantic_summary(spans: dict) -> dict:
    """Build summary statistics for semantic artifact."""
    authority = {"strong": 0, "weak": 0, "none": 0, "missing": 0}
    disposition = {"included": 0, "excluded": 0, "interruption": 0, "missing": 0}
    a2_stats = {"edges": 0, "qualified": 0, "cross_stream": 0}
    tts = {"included": 0, "excluded": 0}
    confidence_sum = 0
    confidence_count = 0

    for cid, sp in spans.items():
        # Authority distribution
        contract = sp.get("_ronc_contract") or {}
        auth = contract.get("authority", "missing")
        authority[auth] = authority.get(auth, 0) + 1

        # Disposition distribution
        disp = sp.get("_semantic_disposition", "missing")
        disposition[disp] = disposition.get(disp, 0) + 1

        # Confidence stats
        conf = sp.get("_semantic_confidence")
        if conf is not None:
            confidence_sum += conf
            confidence_count += 1

        # A2 edge stats
        if sp.get("_a2_edge_exists"):
            a2_stats["edges"] += 1
        if sp.get("_a2_qualified"):
            a2_stats["qualified"] += 1
        if sp.get("_a2_cross_stream"):
            a2_stats["cross_stream"] += 1

        # TTS eligibility
        if sp.get("_tts_excluded"):
            tts["excluded"] += 1
        else:
            tts["included"] += 1

    return {
        "total_spans": len(spans),
        "authority_distribution": authority,
        "disposition_distribution": disposition,
        "a2_edge_stats": a2_stats,
        "tts_eligibility": tts,
        "avg_semantic_confidence": (
            round(confidence_sum / confidence_count, 3)
            if confidence_count > 0 else None
        ),
    }

# ========================================
# STAGE 3: TTS Generation (Unchanged)
# ========================================

async def generate_single_chunk(
        chunk: dict,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        _logger
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


def atomic_write_manifest(path, data, _logger):
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


def validate_and_write_manifest(path, data, trace_id, _logger):
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