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
import base64
import json
from pathlib import Path
import logging
import asyncio
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
import os
from typing import Any, List, Optional, Set
from pydantic import BaseModel, Field
import tempfile
import shutil
import re
import unicodedata
import wave
import io
from collections import Counter

from . import extraction_engine

from ..service_health import build_health_response
from ..artifact_store import (
    build_artifact_ref_from_path,
    document_id_for_file,
    merge_artifact_ref_dicts,
    register_artifact,
    write_document_asset,
    write_bytes_atomic,
)
from ..job_store import (
    append_job_event,
    list_job_events,
    list_jobs,
    load_job,
    manifest_job_fields,
    sync_job_from_manifest,
    update_job_state,
)
from ..toolset_contracts import (
    DocumentAsset,
    audiobook_id_from_document_id,
    model_to_dict,
    new_job_id,
    new_trace_id,
    sha256_file,
    utc_now_iso,
)

logging.getLogger("httpx").setLevel(logging.WARNING)


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
    artifact_ref: Optional[dict] = None


class ManifestSchema(BaseModel):
    # Core Metadata
    metadata: dict = Field(..., description="Original PDF metadata")
    book_id: str
    trace_id: str
    schema_version: str = "tts.audiobook_manifest.v1"
    document_id: Optional[str] = None
    job_id: Optional[str] = None
    audiobook_id: Optional[str] = None

    # State and Progress
    processing_status: Optional[str] = Field(None)
    job_status: Optional[str] = Field(None)
    job_stage: Optional[str] = Field(None)
    progress_current: int = Field(0)
    progress_total: int = Field(0)
    progress_percentage: float = Field(0.0)
    total_chunks: int = Field(0)
    ready_chunks: List[ReadyChunkSchema] = Field(default_factory=list)
    artifact_refs: List[dict] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Error/Recovery
    error_message: Optional[str] = Field(None)
    sentence_index: Optional[dict] = None
    orphaned_audio_artifacts: List[dict] = Field(default_factory=list)
    failed_chunks: List[dict] = Field(default_factory=list)
    stage3_failed_chunks: List[dict] = Field(default_factory=list)
    chunk_state_dir: Optional[str] = None
    stage3_worker_count: Optional[int] = None


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


def _ensure_workspace_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="PDF Processing Service")
client = httpx.AsyncClient(timeout=300.0)

TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8000/api/tts")
DOCTR_SERVICE_URL = os.getenv("DOCTR_SERVICE_URL", "http://host.docker.internal:8004")


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


TTS_MAX_CONCURRENT_REQUESTS = _env_int('TTS_MAX_CONCURRENT_REQUESTS', 10)
TTS_STAGE3_WORKER_COUNT = _env_int(
    'TTS_STAGE3_WORKER_COUNT',
    min(6, TTS_MAX_CONCURRENT_REQUESTS),
    maximum=TTS_MAX_CONCURRENT_REQUESTS,
)
TTS_STAGE3_PROGRESS_EVENT_EVERY = _env_int('TTS_STAGE3_PROGRESS_EVENT_EVERY', 10)
TTS_SEMAPHORE = asyncio.Semaphore(TTS_MAX_CONCURRENT_REQUESTS)
MANIFEST_LOCK = asyncio.Lock()
ARTIFACT_SHARD_DIRNAME = "artifact_shards"
AUDIO_TIMING_FILENAME = "audio_timing.json"

TTS_MAX_CHUNK_CHARS = 650
_TTS_UNIT_MIN_VIABLE_WORDS: int = 3
_TTS_COMPOUND_SEGMENT_MIN_CHARS: int = 3
_TTS_PROACTIVE_SPLIT_CHARS: int = 130
_TTS_PROACTIVE_SPLIT_WORDS: int = 22

# Stage 3 TTS output validation thresholds
_TTS_MAX_WAV_SECONDS: float = 120.0  # Absolute ceiling
_TTS_MAX_DURATION_RATIO: float = 5.0  # Max actual/expected
_TTS_MIN_DURATION_RATIO: float = 0.1  # Min actual/expected (deflation)
_TTS_MIN_EXPECTED_SECONDS: float = 1.5  # Floor for unit expected duration

EXTRACTOR_VERSION = "2.2"

MAX_CONSECUTIVE_FAILURES = 5


def _manifest_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _header_value(request: Request, name: str) -> Optional[str]:
    value = request.headers.get(name)
    value = value.strip() if isinstance(value, str) else None
    return value or None


def _processing_identity_from_request(pdf_path: Path, request: Request) -> dict:
    trace_id = _header_value(request, "X-Trace-ID") or new_trace_id()
    document_id = _header_value(request, "X-Document-ID") or document_id_for_file(pdf_path)
    job_id = _header_value(request, "X-Job-ID") or new_job_id()
    audiobook_id = _header_value(request, "X-Audiobook-ID") or audiobook_id_from_document_id(
        document_id)
    book_id = derive_book_id(audiobook_id)
    return {
        "trace_id": trace_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "book_id": book_id,
    }


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _identity_from_manifest(book_id: str, trace_id: Optional[str] = None) -> dict:
    manifest = _load_manifest(OUTPUT_DIR / book_id / "manifest.json")
    source_filename = manifest.get("metadata", {}).get("source_filename")
    pdf_path = INPUT_DIR / source_filename if source_filename else None
    document_id = manifest.get("document_id")
    if not document_id and pdf_path and pdf_path.exists():
        document_id = document_id_for_file(pdf_path)
    audiobook_id = manifest.get("audiobook_id") or book_id
    return {
        "trace_id": trace_id or manifest.get("trace_id") or new_trace_id(),
        "document_id": document_id,
        "job_id": manifest.get("job_id"),
        "audiobook_id": audiobook_id,
        "book_id": book_id,
    }


def _register_file_artifact(
        *,
        path: Path,
        role: str,
        trace_id: str,
        book_id: Optional[str] = None,
        document_id: Optional[str] = None,
        job_id: Optional[str] = None,
        audiobook_id: Optional[str] = None,
        mime_type: str,
        schema_version: str,
        metadata: Optional[dict] = None,
) -> dict:
    if book_id and (not document_id or not job_id or not audiobook_id):
        ids = _identity_from_manifest(book_id, trace_id)
        document_id = document_id or ids.get("document_id")
        job_id = job_id or ids.get("job_id")
        audiobook_id = audiobook_id or ids.get("audiobook_id")
    ref = build_artifact_ref_from_path(
        path=path,
        role=role,
        trace_id=trace_id,
        job_id=job_id,
        document_id=document_id,
        audiobook_id=audiobook_id,
        mime_type=mime_type,
        schema_version=schema_version,
        metadata=metadata or {},
    )
    register_artifact(
        ref,
        cache_dir=CACHE_DIR,
        audiobook_dir=(OUTPUT_DIR / book_id) if book_id else None,
    )
    return model_to_dict(ref)


def _merge_manifest_artifact_refs(manifest: dict, refs: List[dict]) -> dict:
    if not refs:
        return manifest
    manifest["artifact_refs"] = merge_artifact_ref_dicts(manifest.get("artifact_refs", []), refs)
    return manifest


def _append_manifest_artifact_ref(manifest_path: Path, ref: dict, trace_id: str) -> None:
    manifest = _load_manifest(manifest_path)
    if not manifest:
        return
    _merge_manifest_artifact_refs(manifest, [ref])
    validate_and_write_manifest(manifest_path, manifest, trace_id, logger)


def _register_manifest_artifact(manifest_path: Path, manifest: dict, trace_id: str) -> None:
    if not manifest_path.exists():
        return
    book_id = manifest.get("book_id") or manifest_path.parent.name
    _register_file_artifact(
        path=manifest_path,
        role="audiobook_manifest",
        trace_id=trace_id,
        book_id=book_id,
        document_id=manifest.get("document_id"),
        job_id=manifest.get("job_id"),
        audiobook_id=manifest.get("audiobook_id") or book_id,
        mime_type="application/json",
        schema_version="tts.audiobook_manifest.v1",
        metadata={"processing_status": manifest.get("processing_status")},
    )


def _record_job_stage(
        manifest_path: Path,
        trace_id: str,
        *,
        status: str = "running",
        stage: str,
        message: str | None = None,
        event_type: str = "stage_started",
        progress_current: int | None = None,
        progress_total: int | None = None,
        error: dict | None = None,
        metadata: dict | None = None,
) -> None:
    manifest = _load_manifest(manifest_path)
    job_id = manifest.get("job_id")
    if not job_id:
        return
    try:
        update_job_state(
            cache_dir=CACHE_DIR,
            job_id=job_id,
            trace_id=trace_id or manifest.get("trace_id") or "",
            document_id=manifest.get("document_id"),
            audiobook_id=manifest.get("audiobook_id") or manifest.get("book_id"),
            status=status,
            stage=stage,
            progress_current=progress_current,
            progress_total=progress_total,
            error=error,
            artifacts=manifest.get("artifact_refs") or [],
            metadata_patch={**(metadata or {}), "book_id": manifest.get("book_id")},
            book_id=manifest.get("book_id"),
            event_type=event_type,
            message=message,
        )
    except Exception as exc:
        logger.warning("[%s] Failed to record job stage %s: %s", trace_id, stage, exc)


def _artifact_shard_root(book_id: str, role: str) -> Path:
    safe_book_id = derive_book_id(book_id)
    safe_role = re.sub(r"[^\w\-]+", "_", role).strip("_") or "artifact"
    return CACHE_DIR / ARTIFACT_SHARD_DIRNAME / safe_book_id / safe_role


def _resolve_source_pdf_for_book(safe_book_id: str, manifest_path: Path) -> tuple[str, dict, Path]:
    existing_manifest = _load_manifest(manifest_path)
    source_filename = existing_manifest.get("metadata", {}).get("source_filename")

    if not source_filename:
        for pdf_file in INPUT_DIR.glob("*.pdf"):
            if derive_book_id(pdf_file.stem) == safe_book_id:
                source_filename = pdf_file.name
                break

    if not source_filename:
        candidates = [
            f"{safe_book_id}.pdf",
            f"{safe_book_id.replace('_', ' ')}.pdf",
            f"{safe_book_id.replace('_', '-')}.pdf",
        ]
        for candidate in candidates:
            if (INPUT_DIR / candidate).exists():
                source_filename = candidate
                break

    if not source_filename:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for book_id '{safe_book_id}'. Check INPUT_DIR.",
        )

    source_pdf_path = INPUT_DIR / source_filename
    if not source_pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source PDF '{source_filename}' not found in INPUT_DIR.",
        )

    return str(source_filename), existing_manifest, source_pdf_path


def _clear_book_outputs_and_caches(safe_book_id: str, source_filename: str, trace_id: str) -> None:
    book_dir = OUTPUT_DIR / safe_book_id
    if book_dir.exists():
        shutil.rmtree(book_dir)
        logger.info("[%s] Removed audiobook output directory: %s", trace_id, book_dir)
    book_dir.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(source_filename).stem
    files_to_remove = set(CACHE_DIR.glob(f"{safe_book_id}_*"))
    files_to_remove.update(CACHE_DIR.glob(f"{pdf_stem}_*"))
    for file_path in files_to_remove:
        if not file_path.is_file():
            continue
        try:
            file_path.unlink()
            logger.info("[%s] Removed cache artifact: %s", trace_id, file_path.name)
        except Exception as exc:
            logger.warning("[%s] Failed to remove cache artifact %s: %s", trace_id, file_path.name,
                           exc)

    shard_root = CACHE_DIR / ARTIFACT_SHARD_DIRNAME / safe_book_id
    if shard_root.exists():
        shutil.rmtree(shard_root)
        logger.info("[%s] Removed shard cache directory: %s", trace_id, shard_root)


def _page_number(value: Any) -> Optional[int]:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _write_json_artifact_shard_index(
        *,
        book_id: str,
        role: str,
        schema_version: str,
        index_payload: dict,
        trace_id: str,
        document_id: Optional[str],
        job_id: Optional[str],
        audiobook_id: Optional[str],
) -> dict:
    root = _artifact_shard_root(book_id, role)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    atomic_write_manifest(index_path, index_payload, logger)
    return _register_file_artifact(
        path=index_path,
        role=f"{role}_shard_index",
        trace_id=trace_id,
        book_id=book_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        mime_type="application/json",
        schema_version=schema_version,
        metadata={
            "page_count": len(index_payload.get("pages") or []),
            "shard_dir": str(root),
        },
    )


def _reset_shard_dir(root: Path) -> None:
    cache_root = (CACHE_DIR / ARTIFACT_SHARD_DIRNAME).resolve()
    target = root.resolve()
    try:
        target.relative_to(cache_root)
    except ValueError:
        raise RuntimeError(f"Refusing to reset shard path outside cache shard root: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _write_semantic_page_shards(
        semantic_data: dict,
        *,
        book_id: str,
        trace_id: str,
        document_id: Optional[str],
        job_id: Optional[str],
        audiobook_id: Optional[str],
) -> Optional[dict]:
    spans = semantic_data.get("spans") or {}
    if not isinstance(spans, dict) or not spans:
        return None

    root = _artifact_shard_root(book_id, "semantic")
    _reset_shard_dir(root)

    pages: dict[int, dict[str, dict]] = {}
    for cid, span in spans.items():
        if not isinstance(span, dict):
            continue
        page = _page_number(span.get("page_number"))
        if page is None:
            continue
        pages.setdefault(page, {})[str(cid)] = span

    index_pages: list[dict] = []
    for page in sorted(pages):
        filename = f"page_{page:06d}.json"
        payload = {
            "schema_version": "tts.semantic_page.v1",
            "artifact_type": "semantic_page",
            "book_id": book_id,
            "document_id": document_id,
            "job_id": job_id,
            "audiobook_id": audiobook_id,
            "trace_id": trace_id,
            "page_number": page,
            "generated_at": utc_now_iso(),
            "source_schema_version": semantic_data.get("schema_version"),
            "spans": pages[page],
            "summary": _build_semantic_summary(pages[page]),
        }
        atomic_write_manifest(root / filename, payload, logger)
        index_pages.append({
            "page_number": page,
            "filename": filename,
            "span_count": len(pages[page]),
        })

    index_payload = {
        "schema_version": "tts.semantic_shard_index.v1",
        "artifact_type": "semantic_shard_index",
        "book_id": book_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "trace_id": trace_id,
        "generated_at": utc_now_iso(),
        "source_artifact": f"{book_id}_semantic.json",
        "shard_role": "semantic",
        "pages": index_pages,
        "summary": {
            "total_pages": len(index_pages),
            "total_spans": sum(item["span_count"] for item in index_pages),
        },
    }
    return _write_json_artifact_shard_index(
        book_id=book_id,
        role="semantic",
        schema_version="tts.semantic_shard_index.v1",
        index_payload=index_payload,
        trace_id=trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
    )


def _write_ui_sentence_page_shards(
        ui_data: dict,
        *,
        book_id: str,
        trace_id: str,
        document_id: Optional[str],
        job_id: Optional[str],
        audiobook_id: Optional[str],
) -> Optional[dict]:
    sentences = ui_data.get("sentences") or []
    if not isinstance(sentences, list) or not sentences:
        return None

    root = _artifact_shard_root(book_id, "ui_sentences")
    _reset_shard_dir(root)

    pages: dict[int, list[dict]] = {}
    for sentence in sentences:
        if not isinstance(sentence, dict):
            continue
        sentence_pages = sentence.get("pages") or []
        if not isinstance(sentence_pages, list):
            sentence_pages = [sentence_pages]
        normalized = sorted(
            {page for page in (_page_number(value) for value in sentence_pages) if page})
        for page in normalized:
            pages.setdefault(page, []).append(sentence)

    index_pages: list[dict] = []
    for page in sorted(pages):
        filename = f"page_{page:06d}.json"
        page_sentences = sorted(
            pages[page],
            key=lambda item: (item.get("timing", {}).get("start", 0), item.get("global_index", 0)),
        )
        payload = {
            "schema_version": "tts.ui_sentences_page.v2",
            "artifact_type": "ui_sentences_page",
            "book_id": book_id,
            "document_id": document_id,
            "job_id": job_id,
            "audiobook_id": audiobook_id,
            "trace_id": trace_id,
            "page_number": page,
            "generated_at": utc_now_iso(),
            "source_schema_version": ui_data.get("schema_version"),
            "sentences": page_sentences,
            "summary": {
                "sentence_count": len(page_sentences),
                "cross_page_sentence_count": sum(
                    1 for item in page_sentences if len(item.get("pages") or []) > 1),
            },
        }
        atomic_write_manifest(root / filename, payload, logger)
        index_pages.append({
            "page_number": page,
            "filename": filename,
            "sentence_count": len(page_sentences),
        })

    index_payload = {
        "schema_version": "tts.ui_sentences_shard_index.v2",
        "artifact_type": "ui_sentences_shard_index",
        "book_id": book_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "trace_id": trace_id,
        "generated_at": utc_now_iso(),
        "source_artifact": f"{book_id}_ui_sentences.json",
        "shard_role": "ui_sentences",
        "pages": index_pages,
        "summary": {
            "total_pages": len(index_pages),
            "total_sentences_with_page_membership": sum(
                item["sentence_count"] for item in index_pages),
            "source_total_sentences": ui_data.get("summary", {}).get("total_sentences"),
        },
    }
    return _write_json_artifact_shard_index(
        book_id=book_id,
        role="ui_sentences",
        schema_version="tts.ui_sentences_shard_index.v2",
        index_payload=index_payload,
        trace_id=trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
    )


def _stage3_checkpoint_path(book_id: str) -> Path:
    return OUTPUT_DIR / book_id / "stage3_checkpoint.json"


def _stage3_chunk_checkpoint_dir(book_id: str) -> Path:
    return OUTPUT_DIR / book_id / "stage3_chunks"


def _stage3_chunk_checkpoint_path(book_id: str, chunk_id: int) -> Path:
    return _stage3_chunk_checkpoint_dir(book_id) / f"chunk_{int(chunk_id):06d}.json"


def _chunk_page_list(chunk: dict) -> list[int]:
    values = []
    raw_pages = chunk.get("pages")
    if isinstance(raw_pages, list):
        values.extend(raw_pages)
    elif raw_pages is not None:
        values.append(raw_pages)
    if chunk.get("page") is not None:
        values.append(chunk.get("page"))
    return sorted({page for page in (_page_number(value) for value in values) if page})


def _write_stage3_chunk_checkpoint(
        *,
        book_id: str,
        trace_id: str,
        chunk: dict,
        status: str,
        worker_id: Optional[int] = None,
        error_message: Optional[str] = None,
) -> Optional[dict]:
    chunk_id = _nonnegative_int(chunk.get("chunk_id"))
    if chunk_id is None:
        return None
    ids = _identity_from_manifest(book_id, trace_id)
    page = _page_number(chunk.get("page"))
    audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav" if page is not None else f"chunk_{chunk_id:04d}.wav"
    payload = {
        "schema_version": "tts.stage3_chunk_checkpoint.v1",
        "artifact_type": "stage3_chunk_checkpoint",
        "book_id": book_id,
        "document_id": ids.get("document_id"),
        "job_id": ids.get("job_id"),
        "audiobook_id": ids.get("audiobook_id"),
        "trace_id": trace_id,
        "chunk_id": chunk_id,
        "page": page,
        "pages": _chunk_page_list(chunk),
        "audio_filename": audio_filename,
        "status": status,
        "worker_id": worker_id,
        "error_message": error_message,
        "updated_at": utc_now_iso(),
    }
    if status in {"ready", "failed", "skipped"}:
        payload["completed_at"] = payload["updated_at"]
    checkpoint_path = _stage3_chunk_checkpoint_path(book_id, chunk_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_manifest(checkpoint_path, payload, logger)
    return payload


def _write_stage3_checkpoint(
        *,
        book_id: str,
        trace_id: str,
        total_chunks: int,
        worker_count: int,
        completed_count: int,
        succeeded_chunk_ids: list[int],
        failed_chunks: list[dict],
) -> dict:
    ids = _identity_from_manifest(book_id, trace_id)
    failed_ids = [item.get("chunk_id") for item in failed_chunks if
                  item.get("chunk_id") is not None]
    completed_ids = sorted(
        set(succeeded_chunk_ids + [int(item) for item in failed_ids if isinstance(item, int)]))
    pending_count = max(0, int(total_chunks) - int(completed_count))
    payload = {
        "schema_version": "tts.stage3_checkpoint.v1",
        "artifact_type": "stage3_checkpoint",
        "book_id": book_id,
        "document_id": ids.get("document_id"),
        "job_id": ids.get("job_id"),
        "audiobook_id": ids.get("audiobook_id"),
        "trace_id": trace_id,
        "stage": "stage_3_audio",
        "chunk_state_dir": str(_stage3_chunk_checkpoint_dir(book_id)),
        "worker_count": worker_count,
        "total_chunks": total_chunks,
        "completed_count": completed_count,
        "succeeded_count": len(set(succeeded_chunk_ids)),
        "failed_count": len(failed_chunks),
        "pending_count": pending_count,
        "succeeded_chunk_ids": sorted(set(succeeded_chunk_ids)),
        "failed_chunks": failed_chunks,
        "completed_chunk_ids": completed_ids,
        "updated_at": utc_now_iso(),
    }
    atomic_write_manifest(_stage3_checkpoint_path(book_id), payload, logger)
    return payload


def _register_stage3_checkpoint_artifact(book_id: str, trace_id: str, manifest_path: Path) -> \
        Optional[dict]:
    checkpoint_path = _stage3_checkpoint_path(book_id)
    if not checkpoint_path.exists():
        return None
    ids = _identity_from_manifest(book_id, trace_id)
    ref = _register_file_artifact(
        path=checkpoint_path,
        role="stage3_checkpoint",
        trace_id=trace_id,
        book_id=book_id,
        document_id=ids.get("document_id"),
        job_id=ids.get("job_id"),
        audiobook_id=ids.get("audiobook_id"),
        mime_type="application/json",
        schema_version="tts.stage3_checkpoint.v1",
        metadata={"stage": "stage_3_audio"},
    )
    _append_manifest_artifact_ref(manifest_path, ref, trace_id)
    return ref


def _audio_timing_path(book_id: str) -> Path:
    return OUTPUT_DIR / book_id / AUDIO_TIMING_FILENAME


def _wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = float(wav.getframerate() or 1)
            return frames / rate
    except Exception:
        return None


def _chunk_audio_path(book_id: str, ready_chunk: dict) -> Optional[Path]:
    filename = ready_chunk.get("filename")
    if not filename:
        return None
    candidate = (OUTPUT_DIR / book_id / str(filename)).resolve()
    try:
        candidate.relative_to((OUTPUT_DIR / book_id).resolve())
    except ValueError:
        return None
    return candidate


def _sentence_estimated_duration(sentence: dict) -> float:
    value = sentence.get("duration_seconds")
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    start = sentence.get("start_time")
    end = sentence.get("end_time")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return float(end - start)
    return 0.0


def _write_audio_timing_artifact(
        *,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        citation_json_path: Path,
) -> Optional[dict]:
    """Create measured chunk timing plus proportional sentence timing after Stage 3.

    The current TTS path does not expose true word/sentence timestamps from the
    TTS engine. This artifact therefore uses actual WAV chunk duration as the
    measured authority and proportionally maps sentence windows inside the
    measured chunk using Stage 2 sentence duration estimates. The contract is
    explicit about that basis so downstream UI can prefer it without mistaking it
    for word-level forced alignment.
    """
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[%s] Audio timing skipped; manifest unreadable: %s", trace_id, exc)
        return None

    try:
        citation_data = json.loads(
            citation_json_path.read_text(encoding="utf-8")) if citation_json_path.exists() else {}
    except Exception:
        citation_data = {}

    ids = _identity_from_manifest(book_id, trace_id)
    chunk_source = {int(c.get("chunk_id")): c for c in citation_data.get("chunks", []) if
                    isinstance(c, dict) and c.get("chunk_id") is not None}
    timing_chunks = []
    measured_chunk_count = 0
    proportional_sentence_count = 0

    next_actual_start: Optional[float] = None

    for ready in sorted(manifest.get("ready_chunks", []) or [],
                        key=lambda item: item.get("chunk_id", 0)):
        if not isinstance(ready, dict):
            continue
        chunk_id = _nonnegative_int(ready.get("chunk_id"))
        if chunk_id is None:
            continue
        audio_path = _chunk_audio_path(book_id, ready)
        actual_duration = _wav_duration_seconds(
            audio_path) if audio_path and audio_path.exists() else None
        if actual_duration is not None:
            measured_chunk_count += 1

        estimated_start = float(ready.get("start_time") or 0.0)
        estimated_duration = float(ready.get("duration_seconds") or 0.0)
        estimated_end = float(ready.get("end_time") or (estimated_start + estimated_duration))
        actual_duration_for_timeline = float(
            actual_duration if actual_duration is not None else estimated_duration
        )
        actual_start = estimated_start if next_actual_start is None else next_actual_start
        actual_end = actual_start + max(0.0, actual_duration_for_timeline)
        next_actual_start = actual_end

        source_chunk = chunk_source.get(chunk_id) or ready
        source_sentences = source_chunk.get("sentences") or ready.get("sentences") or []
        sentence_durations = [_sentence_estimated_duration(s) for s in source_sentences if
                              isinstance(s, dict)]
        total_estimated_sentence_duration = sum(v for v in sentence_durations if v > 0)
        actual_cursor = actual_start
        sentences = []

        for idx, sentence in enumerate(source_sentences):
            if not isinstance(sentence, dict):
                continue
            estimated_sentence_duration = _sentence_estimated_duration(sentence)
            if actual_duration is not None and total_estimated_sentence_duration > 0 and estimated_sentence_duration > 0:
                actual_sentence_duration = float(actual_duration) * (
                        estimated_sentence_duration / total_estimated_sentence_duration)
                timing_basis = "measured_chunk_proportional_sentence"
                confidence = "medium"
            else:
                actual_sentence_duration = estimated_sentence_duration
                timing_basis = "estimated_text"
                confidence = "low"

            sentence_actual_start = actual_cursor
            sentence_actual_end = sentence_actual_start + max(0.0, actual_sentence_duration)
            actual_cursor = sentence_actual_end
            proportional_sentence_count += 1
            sentences.append({
                "global_index": sentence.get("global_index"),
                "sentence_in_chunk": sentence.get("sentence_in_chunk", idx),
                "estimated_start": float(sentence.get("start_time") or 0.0),
                "estimated_end": float(sentence.get("end_time") or 0.0),
                "estimated_duration_seconds": estimated_sentence_duration,
                "actual_start": sentence_actual_start,
                "actual_end": sentence_actual_end,
                "actual_duration_seconds": max(0.0, actual_sentence_duration),
                "timing_basis": timing_basis,
                "confidence": confidence,
            })

        timing_chunks.append({
            "chunk_id": chunk_id,
            "filename": ready.get("filename"),
            "pages": ready.get("pages") or (
                [ready.get("page")] if ready.get("page") is not None else []),
            "estimated_start": estimated_start,
            "estimated_end": estimated_end,
            "estimated_duration_seconds": estimated_duration,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "actual_duration_seconds": actual_duration,
            "timing_basis": "measured_wav" if actual_duration is not None else "estimated_text",
            "confidence": "high" if actual_duration is not None else "low",
            "audio_path": str(audio_path) if audio_path else None,
            "sentences": sentences,
        })

    payload = {
        "schema_version": "tts.audio_timing.v1",
        "artifact_type": "audio_timing",
        "book_id": book_id,
        "document_id": ids.get("document_id"),
        "job_id": ids.get("job_id"),
        "audiobook_id": ids.get("audiobook_id"),
        "trace_id": trace_id,
        "generated_at": utc_now_iso(),
        "timing_basis": "measured_wav_chunk_proportional_sentence",
        "coordinate_timebase": "global_audiobook_seconds",
        "timeline_policy": "contiguous_cumulative_chunk_audio",
        "chunks": timing_chunks,
        "summary": {
            "ready_chunk_count": len(manifest.get("ready_chunks", []) or []),
            "timing_chunk_count": len(timing_chunks),
            "measured_chunk_count": measured_chunk_count,
            "proportional_sentence_count": proportional_sentence_count,
        },
    }
    output_path = _audio_timing_path(book_id)
    atomic_write_manifest(output_path, payload, logger)
    ref = _register_file_artifact(
        path=output_path,
        role="audio_timing",
        trace_id=trace_id,
        book_id=book_id,
        document_id=ids.get("document_id"),
        job_id=ids.get("job_id"),
        audiobook_id=ids.get("audiobook_id"),
        mime_type="application/json",
        schema_version="tts.audio_timing.v1",
        metadata=payload["summary"],
    )
    try:
        manifest["audio_timing_ready"] = True
        manifest["audio_timing_artifact_ref"] = ref
        _merge_manifest_artifact_refs(manifest, [ref])
        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
    except Exception as exc:
        logger.warning("[%s] Audio timing artifact written but manifest update failed: %s",
                       trace_id, exc)
    _record_job_stage(
        manifest_path,
        trace_id,
        status=manifest.get("processing_status") or "stage_3_partial",
        stage="stage_3_audio_timing",
        message="Audio timing artifact generated from measured WAV chunk durations.",
        event_type="audio_timing_generated",
        metadata=payload["summary"],
    )
    return ref


def _should_emit_stage3_progress(completed: int, total: int) -> bool:
    if completed <= 0:
        return False
    return completed == total or completed == 1 or completed % TTS_STAGE3_PROGRESS_EVENT_EVERY == 0


async def _generate_audio_chunks_bounded(
        *,
        chunks: list[dict],
        book_id: str,
        trace_id: str,
        manifest_path: Path,
) -> dict:
    """Run Stage 3 with a fixed-size worker pool instead of one task per chunk."""
    total = len(chunks)
    worker_count = min(max(1, TTS_STAGE3_WORKER_COUNT), total)
    queue: asyncio.Queue[tuple[int, dict] | None] = asyncio.Queue(maxsize=max(1, worker_count * 2))
    progress_lock = asyncio.Lock()
    succeeded_chunk_ids: list[int] = []
    failed_chunks: list[dict] = []
    state = {"completed": 0, "successes": 0, "failures": 0}

    try:
        _write_stage3_checkpoint(
            book_id=book_id,
            trace_id=trace_id,
            total_chunks=total,
            worker_count=worker_count,
            completed_count=0,
            succeeded_chunk_ids=succeeded_chunk_ids,
            failed_chunks=failed_chunks,
        )
    except Exception as exc:
        logger.warning("[%s] Failed to write initial Stage 3 checkpoint: %s", trace_id, exc)

    async def worker(worker_id: int) -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                index, chunk = item
                chunk_id = chunk.get("chunk_id", index)
                ok = False
                error_message = None
                try:
                    try:
                        _write_stage3_chunk_checkpoint(
                            book_id=book_id,
                            trace_id=trace_id,
                            chunk=chunk,
                            status="running",
                            worker_id=worker_id,
                        )
                    except Exception as checkpoint_exc:
                        logger.warning(
                            "[%s] Failed to write running checkpoint for chunk %s: %s",
                            trace_id, chunk_id, checkpoint_exc
                        )
                    ok = await generate_single_chunk(chunk, book_id, trace_id, manifest_path,
                                                     logger)
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "[%s] Stage 3 worker %s crashed on chunk %s",
                        trace_id,
                        worker_id,
                        chunk_id,
                        exc_info=True,
                    )

                async with progress_lock:
                    state["completed"] += 1
                    normalized_chunk_id = _nonnegative_int(chunk_id)
                    if normalized_chunk_id is None:
                        normalized_chunk_id = int(index)
                    if ok:
                        state["successes"] += 1
                        succeeded_chunk_ids.append(normalized_chunk_id)
                        try:
                            _write_stage3_chunk_checkpoint(
                                book_id=book_id,
                                trace_id=trace_id,
                                chunk=chunk,
                                status="ready",
                                worker_id=worker_id,
                            )
                        except Exception as checkpoint_exc:
                            logger.warning(
                                "[%s] Failed to write ready checkpoint for chunk %s: %s",
                                trace_id, normalized_chunk_id, checkpoint_exc
                            )
                    else:
                        state["failures"] += 1
                        failure = {
                            "chunk_id": normalized_chunk_id,
                            "worker_id": worker_id,
                            "reason": error_message or "generation_failed",
                        }
                        failed_chunks.append(failure)
                        try:
                            _write_stage3_chunk_checkpoint(
                                book_id=book_id,
                                trace_id=trace_id,
                                chunk=chunk,
                                status="failed",
                                worker_id=worker_id,
                                error_message=failure["reason"],
                            )
                        except Exception as checkpoint_exc:
                            logger.warning(
                                "[%s] Failed to write failed checkpoint for chunk %s: %s",
                                trace_id, normalized_chunk_id, checkpoint_exc
                            )
                    try:
                        _write_stage3_checkpoint(
                            book_id=book_id,
                            trace_id=trace_id,
                            total_chunks=total,
                            worker_count=worker_count,
                            completed_count=state["completed"],
                            succeeded_chunk_ids=succeeded_chunk_ids,
                            failed_chunks=failed_chunks,
                        )
                    except Exception as exc:
                        logger.warning("[%s] Failed to write Stage 3 checkpoint: %s", trace_id, exc)
                    if _should_emit_stage3_progress(state["completed"], total):
                        _record_job_stage(
                            manifest_path, trace_id,
                            status="stage_3_running",
                            stage="stage_3_audio",
                            message=(
                                f"Stage 3 progress: {state['completed']}/{total} chunks "
                                f"({state['successes']} ready, {state['failures']} failed)."
                            ),
                            event_type="stage_3_progress",
                            progress_current=state["completed"],
                            progress_total=total,
                            metadata={
                                "worker_count": worker_count,
                                "succeeded_count": state["successes"],
                                "failed_count": state["failures"],
                            },
                        )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(i + 1)) for i in range(worker_count)]
    for index, chunk in enumerate(chunks):
        await queue.put((index, chunk))
    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)

    _register_stage3_checkpoint_artifact(book_id, trace_id, manifest_path)
    return {
        "total": total,
        "worker_count": worker_count,
        "successes": state["successes"],
        "failures": state["failures"],
        "failed_chunks": failed_chunks,
        "succeeded_chunk_ids": sorted(set(succeeded_chunk_ids)),
    }


def _sync_manifest_job_state(manifest_path: Path, manifest: dict,
                             event_type: Optional[str] = None) -> None:
    if manifest_path.name != "manifest.json":
        return
    try:
        sync_job_from_manifest(
            manifest,
            cache_dir=CACHE_DIR,
            audiobook_dir=manifest_path.parent,
            event_type=event_type,
        )
    except Exception as exc:
        logger.warning(
            "[%s] Failed to sync job state for %s: %s",
            manifest.get("trace_id") or "N/A",
            manifest.get("job_id") or manifest_path,
            exc,
        )


def _with_job_progress_fields(manifest: dict) -> dict:
    payload = dict(manifest or {})
    payload.update(manifest_job_fields(payload))
    return payload


def _seed_document_asset(
        *,
        pdf_path: Path,
        trace_id: str,
        document_id: str,
        job_id: str,
        audiobook_id: str,
        source_filename: str,
        canonical_pdf_ref: dict,
) -> None:
    try:
        asset = DocumentAsset(
            document_id=document_id,
            trace_id=trace_id,
            source_kind="upload_pdf",
            source_filename=source_filename,
            content_sha256=sha256_file(pdf_path),
            canonical_pdf=canonical_pdf_ref,
            metadata={"processor_source_filename": source_filename},
        )
        write_document_asset(asset, cache_dir=CACHE_DIR)
    except Exception as exc:
        logger.warning("[%s] Failed to write document asset %s: %s", trace_id, document_id, exc)


def _build_initial_manifest_for_pdf(pdf_path: Path, source_filename: str, book_id: str,
                                    trace_id: str) -> dict:
    document_id = document_id_for_file(pdf_path)
    audiobook_id = book_id
    job_id = new_job_id()
    canonical_pdf_ref = _register_file_artifact(
        path=pdf_path,
        role="canonical_pdf",
        trace_id=trace_id,
        book_id=book_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        mime_type="application/pdf",
        schema_version="tts.canonical_pdf.v1",
        metadata={"source_filename": source_filename, "source": "processor_direct"},
    )
    _seed_document_asset(
        pdf_path=pdf_path,
        trace_id=trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        source_filename=source_filename,
        canonical_pdf_ref=canonical_pdf_ref,
    )
    return {
        "schema_version": "tts.audiobook_manifest.v1",
        "metadata": {"source_filename": source_filename},
        "book_id": book_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "trace_id": trace_id,
        "processing_status": "processing_started",
        "total_chunks": 0,
        "ready_chunks": [],
        "artifact_refs": [canonical_pdf_ref],
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def _extraction_engine_check() -> str:
    return "ok"


# ========================================
# V1.8: Per-Sentence Audio Synthesis
# ========================================

def _concatenate_wav_with_gaps(wav_segments: List[bytes], gap_ms: int = 40) -> bytes:
    """
    Concatenate WAV audio segments with silence gaps between them.

    Used for prosodic clause splitting: each clause is generated separately
    by TTS and concatenated with micro-gaps (30-50ms) to sound like one
    continuous sentence while avoiding TTS decoder overflow.

    ARCHITECTURAL NOTE:
        This is a Stage 3 (audio generation) concern only.
        It does NOT affect semantic segmentation, citation, or chunk logic.

    Args:
        wav_segments: List of WAV file bytes from individual TTS calls
        gap_ms: Milliseconds of silence between segments (default 40ms)

    Returns:
        Combined WAV file bytes
    """
    if not wav_segments:
        return b''
    if len(wav_segments) == 1:
        return wav_segments[0]

    # Read parameters from first segment
    try:
        with wave.open(io.BytesIO(wav_segments[0]), 'r') as w:
            params = w.getparams()
            sample_rate = w.getframerate()
            sample_width = w.getsampwidth()
            n_channels = w.getnchannels()
    except Exception as e:
        logger.warning(f"Failed to read WAV params: {e}, returning first segment")
        return wav_segments[0]

    # Generate silence gap
    gap_samples = int(sample_rate * gap_ms / 1000)
    silence = b'\x00' * (gap_samples * sample_width * n_channels)

    # Combine all audio data
    combined_frames = []
    for i, segment in enumerate(wav_segments):
        try:
            with wave.open(io.BytesIO(segment), 'r') as w:
                # Enforce parameter consistency across segments
                if (
                        w.getframerate() != sample_rate
                        or w.getsampwidth() != sample_width
                        or w.getnchannels() != n_channels
                ):
                    logger.warning(
                        "WAV segment %d params mismatch (rate=%r width=%r ch=%r) != (rate=%r width=%r ch=%r); skipping",
                        i, w.getframerate(), w.getsampwidth(), w.getnchannels(),
                        sample_rate, sample_width, n_channels
                    )
                    continue
                combined_frames.append(w.readframes(w.getnframes()))
        except Exception as e:
            logger.warning(f"Failed to read WAV segment {i}: {e}, skipping")
            continue

        # Add silence gap after all but last segment
        if i < len(wav_segments) - 1:
            combined_frames.append(silence)

    if not combined_frames:
        return wav_segments[0] if wav_segments else b''

    # If we skipped most segments, fail-open to first segment for safety
    if len(combined_frames) < max(1, len(wav_segments) // 2):
        logger.warning(
            "Most WAV segments were skipped during concatenation (%d/%d kept); returning first segment",
            len(combined_frames), len(wav_segments)
        )
        return wav_segments[0]

    # Write combined WAV
    output = io.BytesIO()
    with wave.open(output, 'w') as w:
        w.setparams(params)
        w.writeframes(b''.join(combined_frames))

    final_bytes = output.getvalue()

    # Validate the combined WAV is parseable (fail-open to first segment)
    try:
        with wave.open(io.BytesIO(final_bytes), 'r') as _w:
            _ = _w.getnframes()
    except Exception as e:
        logger.warning(f"Combined WAV parse failed: {e}; returning first segment")
        return wav_segments[0]

    return final_bytes


# ========================================
# Lifecycle Events
# ========================================

@app.on_event("startup")
async def startup_event():
    _ensure_workspace_dirs()
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


@app.get("/health")
async def health():
    checks = {
        "input_dir": "ok" if INPUT_DIR.exists() else f"error: missing {INPUT_DIR}",
        "cache_dir": "ok" if CACHE_DIR.exists() else f"error: missing {CACHE_DIR}",
        "output_dir": "ok" if OUTPUT_DIR.exists() else f"error: missing {OUTPUT_DIR}",
        "extraction_engine": _extraction_engine_check(),
        "tts_service_url": "ok" if TTS_SERVICE_URL else "error: not configured",
    }
    return build_health_response(
        service="tts_pdf_processor",
        role="pdf_processor_api",
        checks=checks,
        details={
            "input_dir": str(INPUT_DIR),
            "cache_dir": str(CACHE_DIR),
            "output_dir": str(OUTPUT_DIR),
            "tts_service_url": TTS_SERVICE_URL,
            "doctr_service_url": DOCTR_SERVICE_URL,
            "extractor_version": EXTRACTOR_VERSION,
            "stage3_worker_count": TTS_STAGE3_WORKER_COUNT,
            "stage3_progress_event_every": TTS_STAGE3_PROGRESS_EVENT_EVERY,
            "artifact_shard_dir": str(CACHE_DIR / ARTIFACT_SHARD_DIRNAME),
        },
    )


# ========================================
# API Endpoints
# ========================================


@app.post("/api/v1/process/{pdf_filename}")
async def start_pdf_processing(
        pdf_filename: str,
        background_tasks: BackgroundTasks,
        raw_request: Request,
):
    _ensure_workspace_dirs()
    safe_filename = re_sanitize(pdf_filename)
    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    pdf_path = INPUT_DIR / safe_filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")

    ids = _processing_identity_from_request(pdf_path, raw_request)
    trace_id = ids["trace_id"]
    document_id = ids["document_id"]
    job_id = ids["job_id"]
    audiobook_id = ids["audiobook_id"]
    book_id = ids["book_id"]

    audio_dir = OUTPUT_DIR / book_id
    manifest_path = audio_dir / "manifest.json"

    audio_dir.mkdir(parents=True, exist_ok=True)

    canonical_pdf_ref = _register_file_artifact(
        path=pdf_path,
        role="canonical_pdf",
        trace_id=trace_id,
        book_id=book_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        mime_type="application/pdf",
        schema_version="tts.canonical_pdf.v1",
        metadata={"source_filename": safe_filename},
    )
    _seed_document_asset(
        pdf_path=pdf_path,
        trace_id=trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        source_filename=safe_filename,
        canonical_pdf_ref=canonical_pdf_ref,
    )

    initial_manifest = {
        "schema_version": "tts.audiobook_manifest.v1",
        "metadata": {"source_filename": safe_filename},
        "book_id": book_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "trace_id": trace_id,
        "processing_status": "processing_started",
        "total_chunks": 0,
        "ready_chunks": [],
        "artifact_refs": [canonical_pdf_ref],
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }

    try:
        validate_and_write_manifest(manifest_path, initial_manifest, trace_id, logger)
        append_job_event(
            cache_dir=CACHE_DIR,
            job_id=job_id,
            trace_id=trace_id,
            event_type="processor_job_accepted",
            status="running",
            stage="processing_started",
            document_id=document_id,
            audiobook_id=audiobook_id,
            book_id=book_id,
            message="Processor accepted audiobook job.",
            progress_current=0,
            progress_total=0,
            data={"pdf_filename": safe_filename, "book_id": book_id},
        )
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to create initial manifest: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize processing")

    background_tasks.add_task(run_full_pipeline, safe_filename, book_id, trace_id, False)

    return {
        "status": "processing_started",
        "book_id": book_id,
        "trace_id": trace_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "canonical_pdf_artifact_id": canonical_pdf_ref.get("artifact_id"),
    }


@app.post("/api/v1/retry/{book_id}")
async def retry_processing(
        book_id: str,
        background_tasks: BackgroundTasks,
        force_rebuild: bool = False
):
    _ensure_workspace_dirs()
    logger.info("Retry requested for audiobook %s", book_id)

    safe_book_id = re_sanitize(book_id)
    manifest_path = OUTPUT_DIR / safe_book_id / "manifest.json"

    source_filename, existing_manifest, source_pdf_path = _resolve_source_pdf_for_book(safe_book_id,
                                                                                       manifest_path)

    # 3. Status check (only if not force_rebuild and manifest exists)
    if not force_rebuild:
        if not existing_manifest:
            raise HTTPException(
                status_code=404,
                detail=f"No manifest found for '{safe_book_id}'. Use force_rebuild=true."
            )
        if existing_manifest.get('processing_status') not in ['failed', 'stage_3_partial',
                                                              'stage_3_started']:
            raise HTTPException(
                status_code=400,
                detail="Job is not in a retryable state. Use force_rebuild=true."
            )

    generated_trace_id = new_trace_id()
    retry_document_id = (existing_manifest or {}).get("document_id") or document_id_for_file(
        source_pdf_path)
    retry_audiobook_id = (existing_manifest or {}).get("audiobook_id") or safe_book_id
    retry_job_id = new_job_id()

    # 4. Full rebuild replaces generated outputs and derived caches for this audiobook.
    if force_rebuild:
        _clear_book_outputs_and_caches(safe_book_id, source_filename, generated_trace_id)

        canonical_pdf_ref = _register_file_artifact(
            path=source_pdf_path,
            role="canonical_pdf",
            trace_id=generated_trace_id,
            book_id=safe_book_id,
            document_id=retry_document_id,
            job_id=retry_job_id,
            audiobook_id=retry_audiobook_id,
            mime_type="application/pdf",
            schema_version="tts.canonical_pdf.v1",
            metadata={"source_filename": source_filename, "retry_force_rebuild": True},
        )
        manifest = {
            "schema_version": "tts.audiobook_manifest.v1",
            "metadata": {"source_filename": source_filename},
            "book_id": safe_book_id,
            "document_id": retry_document_id,
            "job_id": retry_job_id,
            "audiobook_id": retry_audiobook_id,
            "trace_id": generated_trace_id,
            "processing_status": "processing_started",
            "total_chunks": 0,
            "ready_chunks": [],
            "artifact_refs": [canonical_pdf_ref],
            "error_message": None,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
    else:
        manifest = dict(existing_manifest)
        manifest['processing_status'] = 'processing_started'
        manifest['trace_id'] = generated_trace_id
        manifest['document_id'] = retry_document_id
        manifest['job_id'] = retry_job_id
        manifest['audiobook_id'] = retry_audiobook_id
        manifest['error_message'] = None
        manifest['updated_at'] = utc_now_iso()

    # 6. Write manifest and start pipeline
    validate_and_write_manifest(manifest_path, manifest, generated_trace_id, logger)
    _register_manifest_artifact(manifest_path, manifest, generated_trace_id)

    background_tasks.add_task(
        run_full_pipeline, source_filename, safe_book_id, generated_trace_id, force_rebuild
    )

    return {
        "status": "retry_started",
        "book_id": safe_book_id,
        "trace_id": generated_trace_id,
        "document_id": retry_document_id,
        "job_id": retry_job_id,
        "audiobook_id": retry_audiobook_id,
    }


@app.post("/api/v1/rebuild_selective/{book_id}")
async def rebuild_selective(
        book_id: str,
        background_tasks: BackgroundTasks,
        chunk_ids: Optional[str] = None,
        pages: Optional[str] = None,
):
    """
    Selective rebuild: full extraction and semantic preparation with targeted Stage 3 audio generation.

    Args:
        book_id: Target book identifier (from URL path).
        background_tasks: FastAPI background task runner (injected).
        chunk_ids: Comma-separated chunk IDs, e.g. "0,5,9"
        pages: Comma-separated page numbers, e.g. "1,3" (resolved to chunk IDs after Stage 2)
    """
    _ensure_workspace_dirs()
    logger.info("Selective rebuild requested for audiobook %s", book_id)

    safe_book_id = re_sanitize(book_id)
    manifest_path = OUTPUT_DIR / safe_book_id / "manifest.json"

    # 2. Parse targeting parameters
    target_chunk_ids = None
    target_pages = None

    if chunk_ids:
        try:
            target_chunk_ids = {int(x.strip()) for x in chunk_ids.split(",") if x.strip()}
        except ValueError:
            raise HTTPException(status_code=400,
                                detail="chunk_ids must be comma-separated integers")

    if pages:
        try:
            target_pages = {int(x.strip()) for x in pages.split(",") if x.strip()}
        except ValueError:
            raise HTTPException(status_code=400, detail="pages must be comma-separated integers")

    # Union semantics: if both chunk_ids and pages are provided,
    # the final target set is the union of explicitly named chunks
    # plus all chunks resolved from the named pages.
    if not target_chunk_ids and not target_pages:
        raise HTTPException(
            status_code=400,
            detail="Must specify chunk_ids and/or pages. For full rebuild use retry with force_rebuild=true."
        )

    source_filename, existing_identity_manifest, source_pdf_path = _resolve_source_pdf_for_book(
        safe_book_id, manifest_path
    )

    generated_trace_id = new_trace_id()
    selective_document_id = existing_identity_manifest.get("document_id") or document_id_for_file(
        source_pdf_path)
    selective_audiobook_id = existing_identity_manifest.get("audiobook_id") or safe_book_id
    selective_job_id = new_job_id()

    # 4. Selective rebuild replaces generated outputs and derived caches for this audiobook.
    _clear_book_outputs_and_caches(safe_book_id, source_filename, generated_trace_id)

    # 5. Fresh manifest
    canonical_pdf_ref = _register_file_artifact(
        path=source_pdf_path,
        role="canonical_pdf",
        trace_id=generated_trace_id,
        book_id=safe_book_id,
        document_id=selective_document_id,
        job_id=selective_job_id,
        audiobook_id=selective_audiobook_id,
        mime_type="application/pdf",
        schema_version="tts.canonical_pdf.v1",
        metadata={"source_filename": source_filename, "selective_rebuild": True},
    )
    manifest = {
        "schema_version": "tts.audiobook_manifest.v1",
        "metadata": {"source_filename": source_filename},
        "book_id": safe_book_id,
        "document_id": selective_document_id,
        "job_id": selective_job_id,
        "audiobook_id": selective_audiobook_id,
        "trace_id": generated_trace_id,
        "processing_status": "processing_started",
        "total_chunks": 0,
        "ready_chunks": [],
        "artifact_refs": [canonical_pdf_ref],
        "error_message": None,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    validate_and_write_manifest(manifest_path, manifest, generated_trace_id, logger)
    _register_manifest_artifact(manifest_path, manifest, generated_trace_id)

    # 6. Dispatch selective pipeline
    background_tasks.add_task(
        run_selective_pipeline,
        source_filename,
        safe_book_id,
        generated_trace_id,
        target_chunk_ids,
        target_pages,
    )

    return {
        "status": "selective_rebuild_started",
        "book_id": safe_book_id,
        "trace_id": generated_trace_id,
        "document_id": selective_document_id,
        "job_id": selective_job_id,
        "audiobook_id": selective_audiobook_id,
        "target_chunk_ids": sorted(target_chunk_ids) if target_chunk_ids else None,
        "target_pages": sorted(target_pages) if target_pages else None,
    }


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


@app.get("/api/v1/jobs")
async def list_processing_jobs(
        status: Optional[str] = None,
        book_id: Optional[str] = None,
        limit: int = 100,
):
    return {"jobs": list_jobs(cache_dir=CACHE_DIR, status=status, book_id=book_id, limit=limit)}


@app.get("/api/v1/jobs/{job_id}")
async def get_processing_job(job_id: str):
    job = load_job(job_id, cache_dir=CACHE_DIR)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.get("/api/v1/jobs/{job_id}/events")
async def get_processing_job_events(job_id: str, limit: int = 500):
    if not load_job(job_id, cache_dir=CACHE_DIR):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"job_id": job_id, "events": list_job_events(job_id, cache_dir=CACHE_DIR, limit=limit)}


def _safe_shard_role(role: str) -> str:
    candidate = re.sub(r"[^\w\-]+", "_", role).strip("_")
    if candidate not in {"semantic", "ui_sentences"}:
        raise HTTPException(status_code=404, detail=f"Shard role not found: {role}")
    return candidate


@app.get("/api/v1/audiobooks/{book_id}/stage3-checkpoint")
async def get_stage3_checkpoint(book_id: str):
    safe_book_id = derive_book_id(book_id)
    path = _stage3_checkpoint_path(safe_book_id)
    data = _load_manifest(path)
    if not data:
        raise HTTPException(status_code=404, detail=f"Stage 3 checkpoint not found: {book_id}")
    return data


@app.get("/api/v1/audiobooks/{book_id}/stage3/checkpoint")
async def get_stage3_checkpoint_canonical(book_id: str):
    return await get_stage3_checkpoint(book_id)


@app.get("/api/v1/audiobooks/{book_id}/audio-timing")
async def get_audio_timing_artifact(book_id: str):
    safe_book_id = sanitize_filename(book_id)
    path = _audio_timing_path(safe_book_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Audio timing artifact not found: {book_id}")
    return FileResponse(path, media_type="application/json")


@app.get("/api/v1/audiobooks/{book_id}/stage3/chunks/{chunk_id}")
async def get_stage3_chunk_checkpoint(book_id: str, chunk_id: int):
    safe_book_id = derive_book_id(book_id)
    if chunk_id < 0:
        raise HTTPException(status_code=400, detail="chunk_id must be non-negative")
    path = _stage3_chunk_checkpoint_path(safe_book_id, chunk_id)
    data = _load_manifest(path)
    if not data:
        raise HTTPException(status_code=404,
                            detail=f"Stage 3 chunk checkpoint not found: {book_id}/{chunk_id}")
    return data


@app.get("/api/v1/audiobooks/{book_id}/shards/{role}")
async def get_artifact_shard_index(book_id: str, role: str):
    safe_book_id = derive_book_id(book_id)
    safe_role = _safe_shard_role(role)
    path = _artifact_shard_root(safe_book_id, safe_role) / "index.json"
    data = _load_manifest(path)
    if not data:
        raise HTTPException(status_code=404, detail=f"Shard index not found: {book_id}/{role}")
    return data


@app.get("/api/v1/audiobooks/{book_id}/shards/{role}/pages/{page_number}")
async def get_artifact_shard_page(book_id: str, role: str, page_number: int):
    safe_book_id = derive_book_id(book_id)
    safe_role = _safe_shard_role(role)
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be positive")
    path = _artifact_shard_root(safe_book_id, safe_role) / f"page_{page_number:06d}.json"
    data = _load_manifest(path)
    if not data:
        raise HTTPException(status_code=404,
                            detail=f"Shard page not found: {book_id}/{role}/{page_number}")
    return data


# ========================================
# Pipeline Orchestrator
# ========================================

async def run_full_pipeline(
        pdf_filename: str,
        book_id: str,
        trace_id: str,
        force_rebuild: bool = False
):
    _ensure_workspace_dirs()
    logger.info(f"[{trace_id}] Pipeline started for: {pdf_filename}")

    pdf_path = INPUT_DIR / pdf_filename
    citation_filename = f"{book_id}_citation_ready.json"
    citation_cache_path = CACHE_DIR / citation_filename
    manifest_path = OUTPUT_DIR / book_id / "manifest.json"
    citation_ready_path = citation_cache_path  # Default; overwritten if Stage 2 runs
    m = None  # Initialized; assigned when manifest is read/created
    try:
        # ====================================================================
        # Version-aware cache gate
        # ====================================================================
        use_cache = False

        if not force_rebuild and citation_cache_path.exists():
            # Check 1: Timestamp freshness
            cache_fresh = not (pdf_path.stat().st_mtime > citation_cache_path.stat().st_mtime)
            # Check 2: Extractor version match
            version_match = False
            try:
                with open(citation_cache_path, 'r', encoding='utf-8') as f:
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
            with open(citation_cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not manifest_path.exists():
                m = _build_initial_manifest_for_pdf(pdf_path, pdf_filename, book_id, trace_id)
                validate_and_write_manifest(manifest_path, m, trace_id, logger)
                _register_manifest_artifact(manifest_path, m, trace_id)

            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)

            # Cache hit: reuse citation_ready + update manifest for Stage 2
            total_chunks = data['processing']['total_chunks']

            # Update manifest metadata
            m['metadata'] = data['metadata']
            m['total_chunks'] = total_chunks
            m['processing_status'] = 'stage_2_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        else:
            # Stage 1: Extraction (extraction_engine)
            logger.info(
                f"[{trace_id}] Stage 1: Extraction (extraction_engine v{EXTRACTOR_VERSION})...")
            _record_job_stage(
                manifest_path, trace_id,
                status="running",
                stage="stage_1_extracting",
                message="Stage 1 extraction started.",
                event_type="stage_1_started",
            )
            raw_cache_path = process_pdf(pdf_filename, book_id, trace_id)
            if not raw_cache_path:
                raise RuntimeError("Stage 1 failed")

            if not manifest_path.exists():
                m = _build_initial_manifest_for_pdf(pdf_path, pdf_filename, book_id, trace_id)
                validate_and_write_manifest(manifest_path, m, trace_id, logger)
                _register_manifest_artifact(manifest_path, m, trace_id)

            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)

            m['processing_status'] = 'stage_1_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)

            # Stage 2: Semantic Chunking
            logger.info(f"[{trace_id}] Stage 2: Semantic Chunking...")
            _record_job_stage(
                manifest_path, trace_id,
                status="running",
                stage="stage_2_semantic",
                message="Stage 2 semantic chunking started.",
                event_type="stage_2_started",
            )
            citation_ready_path = prepare_tts_chunks_with_citations(raw_cache_path, trace_id)
            if not citation_ready_path:
                raise RuntimeError("Stage 2 failed")

            # Update Manifest
            with open(citation_ready_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            m['metadata'] = data['metadata']
            m['total_chunks'] = data['processing']['total_chunks']
            m['processing_status'] = 'stage_2_complete'
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        # Stage 3: Audio Generation (ALWAYS RUN)

        logger.info(f"[{trace_id}] Stage 3: Audio Generation...")

        _record_job_stage(
            manifest_path, trace_id,
            status="running",
            stage="stage_3_audio",
            message="Stage 3 audio generation started.",
            event_type="stage_3_started",
        )

        with open(manifest_path, 'r', encoding='utf-8') as f:
            m = json.load(f)

        m['processing_status'] = 'stage_3_started'
        validate_and_write_manifest(manifest_path, m, trace_id, logger)

        stage3_success = await generate_audio_streaming(
            citation_ready_path,
            book_id,
            trace_id,
            manifest_path
        )

        if not stage3_success:
            logger.warning(f"[{trace_id}] Stage 3 completed with failures")

        # Final Status Check with Reconciliation
        final_status = reconcile_manifest_with_disk(book_id, manifest_path, trace_id)
        logger.info(f"[{trace_id}] Job Finished: {final_status}")

    except Exception as e:
        logger.error(f"[{trace_id}] Critical Failure: {e}", exc_info=True)
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            m['processing_status'] = 'failed'
            m['error_message'] = str(e)
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
            _record_job_stage(
                manifest_path, trace_id,
                status="failed",
                stage="failed",
                message=str(e),
                event_type="pipeline_failed",
                error={"message": str(e)},
            )
        except Exception as manifest_err:
            logger.warning(
                f"[{trace_id}] Failed to update manifest after pipeline failure: {manifest_err}")


async def run_selective_pipeline(
        pdf_filename: str,
        book_id: str,
        trace_id: str,
        target_chunk_ids: Optional[Set[int]] = None,
        target_pages: Optional[Set[int]] = None,
):
    """
    Selective pipeline: full Stages 1+2, filtered Stage 3.

    Stages 1, 1.5, 2, 2.5 run identically to run_full_pipeline with force_rebuild=True.
    Stage 3 generates audio only for targeted chunks.
    """
    logger.info(f"[{trace_id}] Selective pipeline started for: {pdf_filename}")

    manifest_path = OUTPUT_DIR / book_id / "manifest.json"
    citation_ready_path = None

    try:
        # ══════════════════════════════════════════════════════════════
        # STAGE 1: Extraction (identical to run_full_pipeline)
        # ══════════════════════════════════════════════════════════════
        logger.info(
            f"[{trace_id}] Stage 1: Extraction (extraction_engine v{EXTRACTOR_VERSION})...")
        _record_job_stage(
            manifest_path, trace_id,
            status="running",
            stage="stage_1_extracting",
            message="Selective rebuild Stage 1 extraction started.",
            event_type="stage_1_started",
            metadata={"selective_rebuild": True},
        )
        raw_cache_path = process_pdf(pdf_filename, book_id, trace_id)
        if not raw_cache_path:
            raise RuntimeError("Stage 1 failed")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        m['processing_status'] = 'stage_1_complete'
        validate_and_write_manifest(manifest_path, m, trace_id, logger)

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Semantic Chunking (identical to run_full_pipeline)
        # ══════════════════════════════════════════════════════════════
        logger.info(f"[{trace_id}] Stage 2: Semantic Chunking...")
        _record_job_stage(
            manifest_path, trace_id,
            status="running",
            stage="stage_2_semantic",
            message="Selective rebuild Stage 2 semantic chunking started.",
            event_type="stage_2_started",
            metadata={"selective_rebuild": True},
        )
        citation_ready_path = prepare_tts_chunks_with_citations(raw_cache_path, trace_id)
        if not citation_ready_path:
            raise RuntimeError("Stage 2 failed")

        with open(citation_ready_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(manifest_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        m['metadata'] = data['metadata']
        m['total_chunks'] = data['processing']['total_chunks']
        m['processing_status'] = 'stage_2_complete'
        validate_and_write_manifest(manifest_path, m, trace_id, logger)

        # ══════════════════════════════════════════════════════════════
        # RESOLVE TARGETING: Merge chunk_ids + pages → final set
        # ══════════════════════════════════════════════════════════════
        resolved_chunk_ids = set(target_chunk_ids) if target_chunk_ids else set()

        if target_pages:
            for chunk in data.get('chunks', []):
                chunk_page = chunk.get('page')
                chunk_pages = chunk.get('pages') or ([chunk_page] if chunk_page is not None else [])
                if any(p in target_pages for p in chunk_pages):
                    resolved_chunk_ids.add(chunk['chunk_id'])
            logger.info(
                f"[{trace_id}] Pages {sorted(target_pages)} resolved; "
                f"targeting chunks: {sorted(resolved_chunk_ids)}"
            )

        if not resolved_chunk_ids:
            logger.warning(
                f"[{trace_id}] No chunks resolved from targeting parameters. "
                f"Stage 3 will produce no audio."
            )

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Audio Generation (SELECTIVE)
        # ══════════════════════════════════════════════════════════════
        logger.info(
            f"[{trace_id}] Stage 3: Selective Audio Generation "
            f"(targeting {len(resolved_chunk_ids)} chunks: {sorted(resolved_chunk_ids)})"
        )

        _record_job_stage(
            manifest_path, trace_id,
            status="running",
            stage="stage_3_audio",
            message="Selective rebuild Stage 3 audio generation started.",
            event_type="stage_3_started",
            metadata={"selective_rebuild": True, "target_chunk_count": len(resolved_chunk_ids)},
        )

        with open(manifest_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        m['processing_status'] = 'stage_3_started'
        validate_and_write_manifest(manifest_path, m, trace_id, logger)

        stage3_success = await generate_audio_streaming(
            citation_ready_path,
            book_id,
            trace_id,
            manifest_path,
            target_chunk_ids=resolved_chunk_ids,
        )

        if not stage3_success:
            logger.warning(f"[{trace_id}] Stage 3 (selective) completed with failures")

        # Final reconciliation
        final_status = reconcile_manifest_with_disk(book_id, manifest_path, trace_id)
        logger.info(f"[{trace_id}] Selective Job Finished: {final_status}")

    except Exception as e:
        logger.error(f"[{trace_id}] Selective Pipeline Failure: {e}", exc_info=True)
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            m['processing_status'] = 'failed'
            m['error_message'] = str(e)
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        except Exception as manifest_err:
            logger.warning(
                f"[{trace_id}] Failed to update manifest after selective failure: {manifest_err}")


# ========================================
# STAGE 1 OCR HELPERS
# ========================================

def _page_has_text_layer(page_data: dict) -> bool:
    """
    Return True if Stage 1 extraction produced any non-empty cleaned text
    for the page's classified spans.
    """
    spans = page_data.get("classified_spans", []) or []
    return any((sp.get("cleaned_text") or "").strip() for sp in spans)


def _extract_doctr_spans_from_response(ocr_payload: dict) -> list:
    """
    Normalize docTR /ocr response into a span list.
    Actual contract: { "spans": [...] }
    Returns [] if no usable span list found.
    """
    if not isinstance(ocr_payload, dict):
        return []
    spans = ocr_payload.get("spans")
    if isinstance(spans, list):
        return spans
    return []


def _route_page_to_doctr(
        doc: "fitz.Document",
        page_num: int,
        trace_id: str = None
) -> list:
    """
    Render doc[page_num] to a PNG image and send to docTR POST /ocr.
    Contract: JSON { image_b64, page_width, page_height, page_num }
    Returns a span list for page_data['classified_spans'], or [] on failure.
    Fails open — pipeline remains stable if docTR is unavailable.
    """
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        payload = {
            "image_b64": base64.b64encode(pix.tobytes("png")).decode("utf-8"),
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "page_num": page_num,
        }

        with httpx.Client(timeout=120.0) as sync_client:
            resp = sync_client.post(
                f"{DOCTR_SERVICE_URL}/ocr",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()

        ocr_spans = _extract_doctr_spans_from_response(result)
        if ocr_spans:
            logger.info(
                f"[{trace_id}] Stage 1 OCR: page {page_num + 1} — "
                f"{len(ocr_spans)} spans returned"
            )
            return ocr_spans

        logger.warning(
            f"[{trace_id}] Stage 1 OCR: page {page_num + 1} — no usable spans in response"
        )
        return []

    except Exception as e:
        logger.warning(
            f"[{trace_id}] Stage 1 OCR: page {page_num + 1} failed — {e}"
        )
        return []


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
                "extractor_version": EXTRACTOR_VERSION
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
                if not _page_has_text_layer(page_data):
                    logger.info(
                        f"[{trace_id}] Stage 1: no text layer on page {page_num + 1} "
                        f"— attempting processor-side OCR"
                    )
                    ocr_spans = _route_page_to_doctr(doc, page_num, trace_id)
                    if ocr_spans:
                        page_data["classified_spans"] = ocr_spans
                page_outputs.append(page_data)

            # Stage 1.5: Normalize headers/footers across document
            extraction_engine.normalize_header_footer_across_document(
                page_outputs,
                trace_id=trace_id
            )
            # Document-scope bibliography/reference role refinement
            extraction_engine.refine_roles_across_document(
                page_outputs,
                trace_id=trace_id
            )

        # Build output structure compatible with Stage 2
        effective_book_id = book_id or derive_book_id(pdf_path.stem)
        ids = _identity_from_manifest(effective_book_id, trace_id)
        metadata.update({
            "document_id": ids.get("document_id"),
            "job_id": ids.get("job_id"),
            "audiobook_id": ids.get("audiobook_id"),
        })
        output_data = {
            "schema_version": "tts.raw_extraction.v1",
            "metadata": metadata,
            "pages": page_outputs,
            "book_id": effective_book_id,
            "document_id": ids.get("document_id"),
            "job_id": ids.get("job_id"),
            "audiobook_id": ids.get("audiobook_id"),
            "trace_id": trace_id,
            "generated_at": utc_now_iso(),
        }

        atomic_write_manifest(cache_file_path, output_data, logger)
        raw_ref = _register_file_artifact(
            path=cache_file_path,
            role="raw_extraction",
            trace_id=trace_id or ids.get("trace_id"),
            book_id=effective_book_id,
            document_id=ids.get("document_id"),
            job_id=ids.get("job_id"),
            audiobook_id=ids.get("audiobook_id"),
            mime_type="application/json",
            schema_version="tts.raw_extraction.v1",
            metadata={"total_pages": metadata.get("total_pages"),
                      "extractor_version": EXTRACTOR_VERSION},
        )
        _append_manifest_artifact_ref(OUTPUT_DIR / effective_book_id / "manifest.json", raw_ref,
                                      trace_id or ids.get("trace_id"))

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

        # ===============================================================
        # CONTRACT VALIDATION: Stage 2 Output Integrity
        # ===============================================================
        if not isinstance(tts_result, dict) or 'chunks' not in tts_result:
            raise RuntimeError("Stage 2 contract violation: missing 'chunks' in tts_result")

        if __debug__:
            for ch in tts_result.get('chunks', []):
                if not isinstance(ch, dict):
                    logger.error("[%s] CONTRACT VIOLATION: non-dict chunk emitted", trace_id)
                    continue
                if 'sentences' not in ch:
                    logger.error(
                        "[%s] CONTRACT VIOLATION: chunk missing sentences: chunk_id=%r",
                        trace_id, ch.get('chunk_id')
                    )

        # Build citation-ready output
        stem = cache_file_path.stem
        if stem.endswith("_raw"):
            stem = stem[:-4]
        book_id = data.get('book_id') or derive_book_id(stem)
        ids = _identity_from_manifest(book_id, trace_id)
        document_id = data.get('document_id') or ids.get("document_id")
        job_id = data.get('job_id') or ids.get("job_id")
        audiobook_id = data.get('audiobook_id') or ids.get("audiobook_id")

        # ═══════════════════════════════════════════════════════════════════
        # SEMANTIC ARTIFACT: Persist RONC/A2/disposition decisions
        # using the Stage 2 processed span authority rather than raw pages.
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

        # ===============================================================
        # DEBUG: Prosodic Metadata Presence Check (belt + suspenders)
        #
        # Check BOTH:
        #   - flattened sentence fields (needs_clause_splitting / prosodic_clauses)
        #   - runtime tracker (sent['_tts_change_tracker'])
        # ===============================================================
        if __debug__:
            for ch in tts_result.get('chunks', []):
                for s in ch.get('sentences', []):
                    needs_split = s.get('needs_clause_splitting', False)
                    clauses = s.get('prosodic_clauses')

                    if needs_split and not clauses:
                        logger.error(
                            "[%s] PROSODY CONTRACT VIOLATION: "
                            "needs_clause_splitting=True but no clauses present",
                            trace_id
                        )

        output_data = {
            'schema_version': 'tts.citation_ready.v1',
            'metadata': data['metadata'],
            'book_id': book_id,
            'document_id': document_id,
            'job_id': job_id,
            'audiobook_id': audiobook_id,
            'trace_id': trace_id,
            'generated_at': utc_now_iso(),
            'processing': tts_result['processing'],
            'document_headers': tts_result.get('document_headers', []),
            'document_footers': tts_result.get('document_footers', []),
            'chunks': tts_result['chunks'],
            'highlighting_enabled': True,
            'page_turn_markers': tts_result.get('page_turn_markers', []),
        }

        atomic_write_manifest(citation_path, output_data, logger)
        citation_ref = _register_file_artifact(
            path=citation_path,
            role="citation_ready",
            trace_id=trace_id,
            book_id=book_id,
            document_id=document_id,
            job_id=job_id,
            audiobook_id=audiobook_id,
            mime_type="application/json",
            schema_version="tts.citation_ready.v1",
            metadata={"total_chunks": tts_result['processing'].get('total_chunks')},
        )
        _append_manifest_artifact_ref(OUTPUT_DIR / book_id / "manifest.json", citation_ref,
                                      trace_id)

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2.5: UI PRESENTATION SYNTHESIS (NON-FATAL)
        # ═══════════════════════════════════════════════════════════════════
        try:
            _generate_ui_sentences_artifact(
                citation_ready_path=citation_path,
                semantic_path=semantic_path,
                book_id=book_id,
                trace_id=trace_id
            )
        except Exception as e:
            logger.warning(
                "[%s] Stage 2.5 UI synthesis failed (non-fatal): %s",
                trace_id, e
            )

        if __debug__ and not citation_path.exists():
            raise RuntimeError("Manifest write failed: citation_ready.json not created")

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
    Formats extraction_engine chunks for manifest and citation artifacts.
    """
    formatted = []
    for chunk in chunks:
        # ===============================================================
        # CONTRACT CHECK: Chunk Structure
        # ===============================================================
        if not isinstance(chunk, dict):
            raise TypeError("Chunk must be a dict")

        required_keys = ("chunk_id", "text", "page", "sentences", "start_time", "end_time")
        for k in required_keys:
            if k not in chunk:
                raise KeyError(f"Chunk missing required key: {k}")

        pages = chunk.get('pages')
        if not pages:
            pages = [chunk.get('page')]
        pages = sorted(p for p in pages if p is not None)

        formatted.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'page': pages[0] if pages else chunk.get('page'),
            'pages': pages,
            'sentences': _format_sentences_for_manifest(chunk['sentences']),
            'start_time': chunk['start_time'],
            'duration_seconds': chunk['duration_seconds'],
            'end_time': chunk['end_time']
        })

    # ===============================================================
    # DEBUG: Sentence Prosody Visibility Check (final line of defense)
    # ===============================================================
    if __debug__:
        for ch in formatted:
            for s in ch.get('sentences', []):
                if s.get('needs_clause_splitting') and not s.get('prosodic_clauses'):
                    logger.error(
                        "MANIFEST CONTRACT VIOLATION: "
                        "needs_clause_splitting=True but prosodic_clauses missing"
                    )

    return formatted


def _format_sentences_for_manifest(sentences: List[dict]) -> List[dict]:
    """
    Formats extraction_engine sentences for manifest artifacts.

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
            # Character offsets for future click-seek precision
            'char_start': sent.get('char_start'),
            'char_end': sent.get('char_end'),
        }

        # ═══════════════════════════════════════════════════════════════════
        # RONC v2.1: Provenance fields (JOIN KEYS — non-negotiable)
        # ═══════════════════════════════════════════════════════════════════

        # Source span canonical IDs (primary join key)
        source_cids = (
                sent.get('source_cids')
                or sent.get('_source_span_ids')
                or sent.get('_source_cids')
                or sent.get('source_span_ids')
        )
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

        # ═══════════════════════════════════════════════════════════════════
        # V1.7: Prosodic Clause Metadata Propagation (TTS Decoder Safety)
        # Prefer flattened sentence fields (survivable), fallback to tracker.
        # ═══════════════════════════════════════════════════════════════════
        needs_split = sent.get('needs_clause_splitting')
        clauses = sent.get('prosodic_clauses')

        if needs_split is None or clauses is None:
            tracker = sent.get('_tts_change_tracker', {}) or {}
            if needs_split is None:
                needs_split = tracker.get('needs_clause_splitting', False)
            if clauses is None:
                clauses = tracker.get('prosodic_clauses', None)

        # Always emit needs_clause_splitting for observability
        entry['needs_clause_splitting'] = bool(needs_split)
        if needs_split:
            entry['prosodic_clauses'] = clauses or []

        formatted.append(entry)

    return formatted


# ========================================
# STAGE 2.5: Backend Contract to Frontend UI Consumption
# ========================================

def _filter_excluded_cids(cids: List[str], span_lookup: dict) -> List[str]:
    """
    Filter CIDs whose spans are marked _tts_excluded.

    Fail-open: CIDs not found in span_lookup are preserved.
    Used by both UI synthesis and citation API to ensure
    highlighting excludes non-narrated content.
    """
    if not span_lookup or not cids:
        return cids
    return [
        cid for cid in cids
        if cid and not span_lookup.get(cid, {}).get("_tts_excluded", False)
    ]


def _generate_ui_sentences_artifact(
        citation_ready_path: Path,
        semantic_path: Path,
        book_id: str,
        trace_id: str = None
) -> Optional[Path]:
    """
    STAGE 2.5: Generate UI presentation synthesis artifact.
    Build the frontend-facing sentence, geometry, and timing artifact from
    citation_ready and semantic artifacts.
    """
    try:
        with open(citation_ready_path, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)

        with open(semantic_path, 'r', encoding='utf-8') as f:
            semantic_data = json.load(f)

        # Prefer the artifact's own book_id over the caller parameter.
        effective_book_id = citation_data.get("book_id") or book_id
        ids = _identity_from_manifest(effective_book_id, trace_id)
        document_id = citation_data.get("document_id") or semantic_data.get(
            "document_id") or ids.get("document_id")
        job_id = citation_data.get("job_id") or semantic_data.get("job_id") or ids.get("job_id")
        audiobook_id = citation_data.get("audiobook_id") or semantic_data.get(
            "audiobook_id") or ids.get("audiobook_id")

        span_lookup: dict[str, dict] = semantic_data.get("spans", {})

        ui_sentences: List[dict] = []
        page_index: dict[str, dict] = {}

        cross_page_count = 0
        stitched_count = 0
        skipped_no_index = 0
        skipped_no_geometry = 0
        all_pages: set = set()

        for chunk in citation_data.get("chunks", []):
            chunk_id = chunk.get("chunk_id")

            for sent in chunk.get("sentences", []):
                global_idx = sent.get("global_index")

                # Guard against missing or invalid global_index.
                if not isinstance(global_idx, int):
                    skipped_no_index += 1
                    if trace_id:
                        logger.warning(
                            "[%s] Stage 2.5: Skipping sentence with invalid "
                            "global_index=%r in chunk %s",
                            trace_id, global_idx, chunk_id
                        )
                    continue

                raw_source_cids = sent.get("source_cids", [])

                # ═══════════════════════════════════════════════════════════════
                # EXCLUSION FILTER: Remove TTS-excluded spans from UI geometry
                # Provenance preserved in citation_ready.json; filtered here for UI
                # ═══════════════════════════════════════════════════════════════
                source_cids = _filter_excluded_cids(raw_source_cids, span_lookup)

                coverage_by_cid = _compute_sentence_cid_coverage(sent, source_cids, span_lookup)

                # Resolve geometry (preserves source_cids order), now carrying
                # backend sentence-ownership ratios for progressive highlighting.
                geometry_by_page = _resolve_geometry_by_page_for_ui(
                    source_cids,
                    span_lookup,
                    coverage_by_cid=coverage_by_cid,
                )

                pages = sorted(int(p) for p in geometry_by_page.keys())

                # Guard: Skip sentences with no resolved geometry (no UI value)
                if not pages:
                    skipped_no_geometry += 1
                    if trace_id:
                        logger.warning(
                            "[%s] Stage 2.5: Sentence %d skipped (no geometry resolved from %d cids)",
                            trace_id, global_idx, len(source_cids)
                        )
                    continue

                all_pages.update(pages)

                # ─────────────────────────────────────────────────────────
                # BUILD UI SENTENCE RECORD
                # V2: explicit display/spoken split, sentence-owned CIDs,
                # coordinate-space/timing declarations, and geometry coverage.
                # ─────────────────────────────────────────────────────────
                spoken_text = sent.get("spoken_text") or sent.get("text") or ""
                display_text = sent.get("display_text") or sent.get("raw_text") or sent.get(
                    "text") or ""
                normalized_text = _normalize_for_alignment(display_text or spoken_text)
                estimated_start = float(sent.get("start_time", 0.0))
                estimated_end = float(sent.get("end_time", 0.0))
                ui_sent: dict[str, Any] = {
                    "global_index": global_idx,
                    "chunk_id": chunk_id,
                    "text": display_text,
                    "display_text": display_text,
                    "spoken_text": spoken_text,
                    "normalized_text": normalized_text,
                    "role": sent.get("role", "body"),
                    "timing": {
                        "start": estimated_start,
                        "end": estimated_end,
                        "estimated_start": estimated_start,
                        "estimated_end": estimated_end,
                        "estimated_duration_seconds": max(0.0, estimated_end - estimated_start),
                        "actual_start": None,
                        "actual_end": None,
                        "actual_duration_seconds": None,
                        "basis": "estimated_text",
                        "confidence": "estimated",
                    },
                    "pages": pages,
                    "cids": source_cids,
                    "source_cids": source_cids,
                    "geometry_cids": source_cids,
                    "geometry": geometry_by_page,
                    "char_start": sent.get("char_start"),
                    "char_end": sent.get("char_end"),
                    "is_stitched": bool(sent.get("is_stitched", False)),
                    "alignment_quality": sent.get("alignment_method", "unknown"),
                    "alignment": {
                        "coverage_method": "backend_text_unique_when_available",
                        "has_backend_coverage": any(
                            (v.get("coverage_method") == "text_unique_alignment")
                            for v in (coverage_by_cid or {}).values()
                        ),
                    },
                }

                # Update counters
                if len(pages) > 1:
                    cross_page_count += 1
                if ui_sent["is_stitched"]:
                    stitched_count += 1

                ui_sentences.append(ui_sent)

                # Build page index
                for page in pages:
                    page_key = str(page)
                    if page_key not in page_index:
                        page_index[page_key] = {"sentence_indices": []}

                    if global_idx not in page_index[page_key]["sentence_indices"]:
                        page_index[page_key]["sentence_indices"].append(global_idx)

        # Sort page_index sentence lists numerically for deterministic UI loads.
        for page_key in page_index:
            page_index[page_key]["sentence_indices"].sort()

        # Attach page turns after sentence construction so from_page is computed
        # from the sentence's own resolved page list.
        page_turn_markers = citation_data.get("page_turn_markers", [])
        _attach_page_turns_to_sentences(ui_sentences, page_turn_markers)

        # ─────────────────────────────────────────────────────────────────
        # ASSEMBLE OUTPUT
        # ═══════════════════════════════════════════════════════════════════
        # ARCHITECTURAL NOTE: excluded_spans intentionally NOT included
        #
        # This artifact serves TTS playback highlighting ONLY (ui_scope).
        # For visible-but-not-narrated content (headers, figures, etc.):
        #   → Load semantic.json, filter spans where _tts_excluded == True
        #
        # This keeps ui_sentences.json single-purpose and avoids
        # data duplication with semantic.json.
        # ─────────────────────────────────────────────────────────────────
        output: dict[str, Any] = {
            "schema_version": "tts.ui_sentences.v2",
            "artifact_type": "ui_sentences",
            "ui_scope": "tts_playback",
            "coordinate_space": {
                "units": "pdf_points",
                "origin": "top_left",
                "bbox_format": "[x0, y0, x1, y1]",
                "page_rotation_applied": False,
                "geometry_authority": "backend_ui_sentences",
            },
            "timing_contract": {
                "timebase": "global_audiobook_seconds",
                "default_basis": "estimated_text",
                "actual_timing_artifact": AUDIO_TIMING_FILENAME,
            },
            "book_id": effective_book_id,
            "document_id": document_id,
            "job_id": job_id,
            "audiobook_id": audiobook_id,
            "trace_id": trace_id,
            "generated_at": utc_now_iso(),
            "source_artifacts": {
                "citation_ready": citation_ready_path.name,
                "semantic": semantic_path.name
            },
            "sentences": ui_sentences,
            "page_index": page_index,
            "summary": {
                "total_sentences": len(ui_sentences),
                "total_pages": len(all_pages),
                "cross_page_sentences": cross_page_count,
                "stitched_sentences": stitched_count,
                "skipped_invalid_index": skipped_no_index,
                "skipped_no_geometry": skipped_no_geometry
            }
        }

        output_path = citation_ready_path.parent / f"{effective_book_id}_ui_sentences.json"
        atomic_write_manifest(output_path, output, logger)
        ui_ref = _register_file_artifact(
            path=output_path,
            role="ui_sentences",
            trace_id=trace_id,
            book_id=effective_book_id,
            document_id=document_id,
            job_id=job_id,
            audiobook_id=audiobook_id,
            mime_type="application/json",
            schema_version="tts.ui_sentences.v2",
            metadata={"total_sentences": len(ui_sentences), "total_pages": len(all_pages)},
        )
        shard_ref = _write_ui_sentence_page_shards(
            output,
            book_id=effective_book_id,
            trace_id=trace_id,
            document_id=document_id,
            job_id=job_id,
            audiobook_id=audiobook_id,
        )
        _append_manifest_artifact_ref(OUTPUT_DIR / effective_book_id / "manifest.json", ui_ref,
                                      trace_id)
        if shard_ref:
            _append_manifest_artifact_ref(OUTPUT_DIR / effective_book_id / "manifest.json",
                                          shard_ref, trace_id)

        logger.info(
            "[%s] Stage 2.5 complete: %d sentences, %d pages, "
            "%d cross-page, %d stitched, %d skipped (idx), %d skipped (geom)",
            trace_id,
            len(ui_sentences),
            len(all_pages),
            cross_page_count,
            stitched_count,
            skipped_no_index,
            skipped_no_geometry
        )

        return output_path

    except Exception as e:
        logger.error(
            "[%s] Stage 2.5 failed (non-fatal, TTS continues): %s",
            trace_id, e,
            exc_info=True
        )
        return None


def _attach_page_turns_to_sentences(
        ui_sentences: List[dict],
        page_turn_markers: List[dict]
) -> None:
    """
    Attach page turns to sentences with correctly computed from_page.

    Computes from_page by examining the sentence's own pages list,
    rather than assuming to_page - 1.

    Mutates ui_sentences in place.

    Args:
        ui_sentences: List of UISentence dicts (must have 'pages' populated)
        page_turn_markers: Raw markers from citation_ready.json
    """
    # Index sentences by global_index for O(1) lookup
    sent_by_idx: dict[int, dict] = {
        s["global_index"]: s for s in ui_sentences
    }

    for marker in page_turn_markers:
        sent_idx = marker.get("sentence_global_index")
        to_page = marker.get("page")
        turn_time = marker.get("turn_time")

        if sent_idx is None or to_page is None:
            continue

        sent = sent_by_idx.get(sent_idx)
        if not sent:
            continue

        # ─────────────────────────────────────────────────────────────────
        # Compute from_page from sentence's pages list
        # Find the page in sentence.pages that precedes to_page
        # NOTE: from_page may be null if to_page not in sentence.pages
        #       (can occur with resolution failures or marker edge cases)
        # ─────────────────────────────────────────────────────────────────
        sentence_pages = sent.get("pages", [])
        from_page = None

        if to_page in sentence_pages:
            # Find the page immediately before to_page in the sentence's page list
            idx = sentence_pages.index(to_page)
            if idx > 0:
                from_page = sentence_pages[idx - 1]

        # Attach page turn
        sent["page_turn"] = {
            "from_page": from_page,  # May be None if not determinable
            "to_page": to_page,
            "turn_time": float(turn_time) if turn_time is not None else 0.0
        }


def _normalize_for_alignment(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sentence_alignment_text(sentence: dict) -> str:
    for key in ("display_text", "text", "spoken_text", "tts_text"):
        value = sentence.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _compute_sentence_cid_coverage(sentence: dict, source_cids: List[str],
                                   span_lookup: dict[str, dict]) -> dict[str, dict]:
    """Return per-CID sentence ownership ratios for UI highlight geometry.

    This is intentionally conservative. When the sentence cannot be uniquely
    aligned within concatenated source span text, every CID remains full-width
    and the method is marked non-authoritative rather than inventing precision.
    """
    coverage = {
        cid: {
            "coverage_start_ratio": 0.0,
            "coverage_end_ratio": 1.0,
            "char_start": None,
            "char_end": None,
            "coverage_method": "full_span_default",
            "coverage_confidence": "low",
        }
        for cid in source_cids
        if cid
    }
    if not source_cids or not span_lookup:
        return coverage

    pieces = []
    combined = ""
    for cid in source_cids:
        span = span_lookup.get(cid) or {}
        cleaned = _normalize_for_alignment(span.get("cleaned_text") or span.get("raw_text") or "")
        start = len(combined)
        combined += cleaned
        end = len(combined)
        combined += " "
        pieces.append({"cid": cid, "text": cleaned, "start": start, "end": end})
    combined = combined.rstrip()
    needle = _normalize_for_alignment(_sentence_alignment_text(sentence))
    if len(needle) < 3 or not combined:
        return coverage

    match = combined.find(needle)
    if match < 0:
        return coverage
    if combined.find(needle, match + 1) >= 0:
        return coverage

    sentence_start = match
    sentence_end = match + len(needle)
    for piece in pieces:
        cid = piece["cid"]
        text_value = piece["text"]
        length = len(text_value)
        if length <= 0:
            coverage[cid] = {
                "coverage_start_ratio": 0.0,
                "coverage_end_ratio": 0.0,
                "char_start": 0,
                "char_end": 0,
                "coverage_method": "text_unique_empty_span",
                "coverage_confidence": "medium",
            }
            continue
        overlap_start = max(piece["start"], sentence_start)
        overlap_end = min(piece["end"], sentence_end)
        if overlap_end <= overlap_start:
            coverage[cid] = {
                "coverage_start_ratio": 0.0,
                "coverage_end_ratio": 0.0,
                "char_start": 0,
                "char_end": 0,
                "coverage_method": "text_unique_outside_sentence",
                "coverage_confidence": "medium",
            }
            continue
        local_start = max(0, overlap_start - piece["start"])
        local_end = max(local_start, overlap_end - piece["start"])
        coverage[cid] = {
            "coverage_start_ratio": round(max(0.0, min(1.0, local_start / length)), 6),
            "coverage_end_ratio": round(max(0.0, min(1.0, local_end / length)), 6),
            "char_start": int(local_start),
            "char_end": int(local_end),
            "coverage_method": "text_unique_alignment",
            "coverage_confidence": "high",
        }
    return coverage


def _resolve_geometry_by_page_for_ui(
        source_cids: List[str],
        span_lookup: dict[str, dict],
        coverage_by_cid: Optional[dict[str, dict]] = None
) -> dict[str, List[dict]]:
    """
    Resolve CID list to geometry grouped by page.

    INVARIANT: Output order within each page preserves source_cids order.
    This is critical for deterministic highlighting.

    Args:
        source_cids: Ordered list of canonical span IDs
        span_lookup: CID → span data mapping from semantic.json

    Returns:
        dict mapping page number (as string) to list of {cid, bbox} dicts.
        Example: {"1": [{"cid": "P0:2", "bbox": [...]}, ...]}
    """
    geometry_by_page: dict[str, List[dict]] = {}

    # Track seen CIDs to enforce uniqueness within sentence
    seen_cids: set = set()

    for cid in source_cids:
        # Skip duplicates (invariant I4)
        if cid in seen_cids:
            continue
        seen_cids.add(cid)

        # Lookup span data
        span = span_lookup.get(cid)
        if not span:
            continue

        # Extract required fields
        page = span.get("page_number")
        bbox = span.get("bbox")

        # Skip if missing required data
        if page is None or bbox is None:
            continue

        # Validate bbox structure
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        # Group by page (preserving insertion order = source_cids order)
        page_key = str(page)
        if page_key not in geometry_by_page:
            geometry_by_page[page_key] = []

        geom_entry = {
            "cid": cid,
            "bbox": list(bbox[:4])  # Ensure list, take first 4 elements
        }
        if coverage_by_cid and cid in coverage_by_cid:
            geom_entry.update(coverage_by_cid[cid])
        geometry_by_page[page_key].append(geom_entry)

    return geometry_by_page


def validate_ui_sentences_contract(
        ui_data: dict[str, Any],
        semantic_data: Optional[dict[str, Any]] = None
) -> List[str]:
    """
    Validate ui_sentences.json against schema invariants.

    Args:
        ui_data: Parsed ui_sentences.json content
        semantic_data: Optional parsed semantic.json for cross-validation

    Returns:
        List of violation descriptions. Empty list = valid.
    """

    errors: List[str] = []

    # ─────────────────────────────────────────────────────────────────────
    # TOP-LEVEL STRUCTURE
    # ─────────────────────────────────────────────────────────────────────
    required_top_keys = [
        "schema_version", "artifact_type", "book_id",
        "sentences", "page_index", "summary"
    ]
    for key in required_top_keys:
        if key not in ui_data:
            errors.append(f"Missing required top-level key: {key}")

    if ui_data.get("artifact_type") != "ui_sentences":
        errors.append(
            f"Invalid artifact_type: {ui_data.get('artifact_type')} "
            f"(expected 'ui_sentences')"
        )

    sentences = ui_data.get("sentences", [])
    page_index = ui_data.get("page_index", {})

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I1: Geometry Order (enabled by 'cids' field)
    # ─────────────────────────────────────────────────────────────────────
    if semantic_data:
        span_lookup = semantic_data.get("spans", {})

        for sent in sentences:
            idx = sent.get("global_index", "?")
            cids = sent.get("cids", [])
            geometry = sent.get("geometry", {})

            for page_key, page_geom in geometry.items():
                expected_cids_for_page = []
                for cid in cids:
                    span = span_lookup.get(cid, {})
                    if str(span.get("page_number")) == page_key:
                        expected_cids_for_page.append(cid)

                actual_cids_for_page = [g.get("cid") for g in page_geom]

                if expected_cids_for_page != actual_cids_for_page:
                    errors.append(
                        f"I1 VIOLATION: Sentence {idx} page {page_key} "
                        f"geometry order mismatch. "
                        f"Expected: {expected_cids_for_page}, "
                        f"Actual: {actual_cids_for_page}"
                    )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I2: Page Consistency
    # ─────────────────────────────────────────────────────────────────────
    for sent in sentences:
        idx = sent.get("global_index", "?")
        pages = set(str(p) for p in sent.get("pages", []))
        geometry_pages = set(sent.get("geometry", {}).keys())

        if geometry_pages - pages:
            errors.append(
                f"I2 VIOLATION: Sentence {idx} has geometry for pages "
                f"{geometry_pages - pages} not in pages list {pages}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I3: Timing Validity
    # ─────────────────────────────────────────────────────────────────────
    for sent in sentences:
        idx = sent.get("global_index", "?")
        timing = sent.get("timing", {})
        start = timing.get("start", 0)
        end = timing.get("end", 0)

        if end <= start:
            errors.append(
                f"I3 VIOLATION: Sentence {idx} has invalid timing "
                f"(start={start}, end={end})"
            )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I4: CID Uniqueness (O(n) using Counter)
    # ─────────────────────────────────────────────────────────────────────
    for sent in sentences:
        idx = sent.get("global_index", "?")
        all_cids = []
        for page_geom in sent.get("geometry", {}).values():
            for entry in page_geom:
                all_cids.append(entry.get("cid"))

        cid_counts = Counter(all_cids)
        duplicates = [cid for cid, count in cid_counts.items() if count > 1]

        if duplicates:
            errors.append(
                f"I4 VIOLATION: Sentence {idx} has duplicate CIDs: {duplicates}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I5: Page Index Completeness
    # ─────────────────────────────────────────────────────────────────────
    for sent in sentences:
        idx = sent.get("global_index", "?")
        for page in sent.get("pages", []):
            page_key = str(page)
            page_entry = page_index.get(page_key, {})
            indices = page_entry.get("sentence_indices", [])

            if idx not in indices:
                errors.append(
                    f"I5 VIOLATION: Sentence {idx} on page {page} "
                    f"not in page_index[{page_key}]"
                )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I6: No Exclusion Leakage (requires semantic_data)
    # ─────────────────────────────────────────────────────────────────────
    if semantic_data:
        span_lookup = semantic_data.get("spans", {})

        for sent in sentences:
            idx = sent.get("global_index", "?")
            for page_geom in sent.get("geometry", {}).values():
                for entry in page_geom:
                    cid = entry.get("cid")
                    span = span_lookup.get(cid, {})

                    if span.get("_tts_excluded", False):
                        errors.append(
                            f"I6 VIOLATION: Sentence {idx} contains "
                            f"excluded span {cid}"
                        )

    # ─────────────────────────────────────────────────────────────────────
    # INVARIANT I7: cids field matches geometry CIDs
    # ─────────────────────────────────────────────────────────────────────
    for sent in sentences:
        idx = sent.get("global_index", "?")
        cids = set(sent.get("cids", []))
        geometry_cids = set()
        for page_geom in sent.get("geometry", {}).values():
            for entry in page_geom:
                geometry_cids.add(entry.get("cid"))

        extra_in_geometry = geometry_cids - cids
        if extra_in_geometry:
            errors.append(
                f"I7 VIOLATION: Sentence {idx} geometry contains CIDs "
                f"not in cids list: {extra_in_geometry}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # EXCLUSION LIST: Forbidden fields
    # ─────────────────────────────────────────────────────────────────────
    forbidden_fields = [
        "_ronc_contract", "_semantic_confidence", "_semantic_disposition",
        "_tts_exclude_reason", "_source_span_ids", "cleaned_text", "raw_text",
        "span_start_index", "span_end_index"
    ]

    for sent in sentences:
        idx = sent.get("global_index", "?")

        for field in forbidden_fields:
            if field in sent:
                errors.append(
                    f"EXCLUSION VIOLATION: Sentence {idx} contains "
                    f"forbidden field '{field}'"
                )

        underscore_fields = [k for k in sent.keys() if k.startswith("_")]
        for field in underscore_fields:
            errors.append(
                f"EXCLUSION VIOLATION: Sentence {idx} contains "
                f"diagnostic field '{field}'"
            )

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY VALIDATION
    # ─────────────────────────────────────────────────────────────────────
    summary = ui_data.get("summary", {})

    if summary.get("total_sentences") != len(sentences):
        errors.append(
            f"Summary mismatch: total_sentences={summary.get('total_sentences')} "
            f"but actual={len(sentences)}"
        )

    actual_cross_page = sum(1 for s in sentences if len(s.get("pages", [])) > 1)
    if summary.get("cross_page_sentences") != actual_cross_page:
        errors.append(
            f"Summary mismatch: cross_page_sentences="
            f"{summary.get('cross_page_sentences')} but actual={actual_cross_page}"
        )

    actual_stitched = sum(1 for s in sentences if s.get("is_stitched"))
    if summary.get("stitched_sentences") != actual_stitched:
        errors.append(
            f"Summary mismatch: stitched_sentences="
            f"{summary.get('stitched_sentences')} but actual={actual_stitched}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # PAGE INDEX SORTING
    # ─────────────────────────────────────────────────────────────────────
    for page_key, page_entry in page_index.items():
        indices = page_entry.get("sentence_indices", [])
        if indices != sorted(indices):
            errors.append(
                f"page_index[{page_key}].sentence_indices is not sorted"
            )

    return errors


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

    Receives the processed_spans dict keyed by canonical span ID from
    compile_tts_ready_content.

    Schema v1.0 Contract:
        - extraction artifact = geometry + basic classification (Stage 1)
        - semantic artifact = RONC authority + dispositions (Stage 2)
        - manifest = final TTS output (Stage 3)

    Scope clarification:
        - _tts_excluded is a Stage 2 eligibility decision
        - manifest.json is the emission authority (Stage 3)
    """
    ids = _identity_from_manifest(book_id, trace_id)
    document_id = metadata.get("document_id") or ids.get("document_id")
    job_id = metadata.get("job_id") or ids.get("job_id")
    audiobook_id = metadata.get("audiobook_id") or ids.get("audiobook_id")
    semantic_data = {
        "schema_version": "tts.semantic.v1",
        "metadata": {
            **metadata,
            "artifact_type": "semantic",
            "schema_version": "tts.semantic.v1",
            "stage": "2",
            "authority_scope": "semantic_eligibility",
        },
        "book_id": book_id,
        "document_id": document_id,
        "job_id": job_id,
        "audiobook_id": audiobook_id,
        "trace_id": trace_id,
        "generated_at": utc_now_iso(),
        "spans": {}
    }

    # Iterate processed_spans dict directly; it is already keyed by CID.
    for cid, sp in processed_spans.items():
        if not cid:
            continue

        semantic_data["spans"][cid] = {
            # RONC Contract (full authority record)
            "_ronc_contract": sp.get("_ronc_contract"),
            # Canonical ID required for provenance tracking
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
            # RONC source fields
            "_ronc_atomic_unit_id": sp.get("_ronc_atomic_unit_id"),
            "_ronc_atomic_role": sp.get("_ronc_atomic_role"),
            "_ronc_break_after": sp.get("_ronc_break_after"),
            "_ronc_rescue_applied": sp.get("_ronc_rescue_applied"),
            # Structural Context (for diagnostics)
            "layout_stream": sp.get("layout_stream"),
            "role": sp.get("role"),
            "page_number": sp.get("page_number"),
            "block_id": sp.get("block_id"),
            "cleaned_text": sp.get("cleaned_text") or "",
            "cleaned_text_preview": (sp.get("cleaned_text") or "")[:100],
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
            # Same-line promotion audit
            # ─────────────────────────────────────────────────────
            "_same_line_promoted": sp.get("_same_line_promoted"),
            "_zombie_role_fixed": sp.get("_zombie_role_fixed"),
            "_original_role": sp.get("_original_role"),
            "_original_layout_stream": sp.get("_original_layout_stream"),
        }

    # ------------------------------------------------------------------
    # Bibliography detection audit (non-behavioral, diagnostics only)
    # ------------------------------------------------------------------
    bibliography_spans = [
        cid
        for cid, sp in semantic_data["spans"].items()
        if sp.get("_tts_exclude_reason") == "role:footnote"
    ]

    if bibliography_spans:
        semantic_data["bibliography_audit"] = {
            "detected_count": len(bibliography_spans),
            "span_ids": bibliography_spans,
        }

    semantic_data["summary"] = _build_semantic_summary(semantic_data["spans"])
    atomic_write_manifest(output_path, semantic_data, logger)
    semantic_ref = _register_file_artifact(
        path=output_path,
        role="semantic",
        trace_id=trace_id,
        book_id=book_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
        mime_type="application/json",
        schema_version="tts.semantic.v1",
        metadata={"span_count": len(semantic_data["spans"])},
    )
    shard_ref = _write_semantic_page_shards(
        semantic_data,
        book_id=book_id,
        trace_id=trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
    )
    _append_manifest_artifact_ref(OUTPUT_DIR / book_id / "manifest.json", semantic_ref, trace_id)
    if shard_ref:
        _append_manifest_artifact_ref(OUTPUT_DIR / book_id / "manifest.json", shard_ref, trace_id)
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
# STAGE 3: TTS Generation
# ========================================
def _prepare_synthesis_units(
        sent: dict,
        trace_id: str = None,
        chunk_id: int = None,
        sent_idx: int = None,
) -> List[str]:
    """
    Pre-TTS gate: Transform sentence into TTS-safe synthesis units.

    Uses existing flags and constants only:
        - needs_clause_splitting (from Stage 2)
        - prosodic_clauses (from Stage 2)
        - TTS_MAX_CHUNK_CHARS (existing safety constant)

    Returns:
        List[str]: Synthesis units ready for individual TTS calls
    """
    sent_text = sent.get('text', '')
    if not sent_text or not sent_text.strip():
        return []

    needs_split = sent.get('needs_clause_splitting', False)
    prosodic_clauses = sent.get('prosodic_clauses')

    # ── PATH 1: Determine initial synthesis units ──
    # Stage 2 flags are advisory, not exclusive. Always allow Stage 3
    # semantic safety logic to assess the sentence for TTS viability.
    if not needs_split or not prosodic_clauses or len(prosodic_clauses) <= 1:
        synthesis_units = [sent_text.strip()]
    else:
        # ── PATH 2: Start with prosodic_clauses, further split if needed ──
        synthesis_units = []

        for clause_idx, clause in enumerate(prosodic_clauses):
            if not clause or not clause.strip():
                continue

            clause = clause.strip()

            # Check against existing safety constant
            if len(clause) <= TTS_MAX_CHUNK_CHARS:
                synthesis_units.append(clause)
            else:
                # Further soft-split this oversized clause
                sub_units = _soft_split_unit(clause, trace_id) or []
                synthesis_units.extend(sub_units)

                if trace_id:
                    logger.debug(
                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "
                        f"clause {clause_idx} further split into {len(sub_units)} units"
                    )

    # ────────────────────────────────────────────────────────────
    # ITERATIVE PROSODIC CONVERGENCE (WORK-QUEUE)
    #
    # Apply safety gates repeatedly until the unit set stabilizes.
    # Convergence is guaranteed by _soft_split_unit contract:
    # it returns unchanged when no split boundary exists.
    #
    # Gates:
    #   1. Hard length (existing constant)
    #   2. Punctuation density (existing heuristic)
    #   3. Structural dominance (sentence-relative, non-arbitrary)
    #
    # Note: units accepted into final_units are finalized; only newly
    # produced sub-units are re-evaluated in subsequent passes.
    # ────────────────────────────────────────────────────────────
    sentence_len = len(sent_text) if sent_text else 0

    work_queue = [u.strip() for u in synthesis_units if u and u.strip()]
    final_units = []

    while work_queue:
        next_queue = []

        for unit in work_queue:
            unit = unit.strip()
            if not unit:
                continue

            unit_len = len(unit)
            unit_word_count = len(unit.split())
            comma_count = unit.count(',')
            semicolon_count = unit.count(';')

            exceeds_length = unit_len > TTS_MAX_CHUNK_CHARS

            # Boundary evidence: if a unit contains a natural split marker, explore splitting.
            has_punct_boundary = (comma_count >= 1 or semicolon_count >= 1)

            # Dominance should apply to *remainders*, not the whole sentence.
            # Prevents "every sentence always attempts split" when starting from sent_text.
            structurally_dominant = (
                    sentence_len > 0 and
                    unit_len < sentence_len and
                    unit_len >= sentence_len // 2
            )

            # ── Clause Structure Classification ──
            # Strong clause structure: multiple commas or semicolons
            # provide reliable L1 delimiter split points. L1 will
            # produce meaningful clause-level units.
            # Weak clause structure: 0-1 commas, no semicolons.
            # A single comma followed by a conjunction ("period, and")
            # can be legally collapsed by L1's conjunction-aware merge,
            # returning the input unchanged.
            has_strong_clause_structure = (
                    comma_count >= 2 or
                    semicolon_count >= 1
            )

            # ── Monotone Length Risk ──
            # Long units without strong clause structure risk stochastic
            # decoder drift: first attempt produces runaway (6-33x ratio),
            # retry with different decoder seed usually succeeds. Each
            # failure costs 30-75s (full decode to max_decoder_steps).
            #
            # Pre-emptive midpoint splitting feeds into existing L2
            # strategy in _soft_split_unit → viability gate validates
            # output. Colons excluded because they provide decoder
            # attention anchoring and L1 does not split on them
            # (midpoint would cross the colon boundary).
            #
            # Work queue cascade naturally re-evaluates halves:
            # structurally_dominant catches halves ≥ sentence_len//2,
            # producing quarters. Entry guard in _soft_split_unit stops
            # further splitting below min_words_for_split threshold.
            # Maximum cascade depth: 2 (original → halves → quarters).
            monotone_length_risk = (
                    not has_strong_clause_structure and
                    ':' not in unit and
                    (
                            unit_len > _TTS_PROACTIVE_SPLIT_CHARS or
                            unit_word_count > _TTS_PROACTIVE_SPLIT_WORDS
                    )
            )

            needs_split_attempt = (
                    exceeds_length or
                    has_punct_boundary or
                    structurally_dominant or
                    monotone_length_risk
            )

            if needs_split_attempt:
                sub_units = _soft_split_unit(unit, trace_id) or []
                sub_units = [x.strip() for x in sub_units if x and x.strip()]

                split_occurred = (
                        len(sub_units) > 1 or
                        (sub_units and sub_units[0] != unit)
                )

                if split_occurred:
                    next_queue.extend(sub_units)
                    if trace_id:
                        logger.debug(
                            f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "
                            f"unit split → {len(sub_units)} sub-units "
                            f"(dom={structurally_dominant}, punct={has_punct_boundary}, mono={monotone_length_risk})"
                        )
                else:
                    final_units.append(unit)
            else:
                final_units.append(unit)

        work_queue = next_queue

    # ────────────────────────────────────────────────────────────
    # ORDER PRESERVATION: Sort units by position in original text
    #
    # The work-queue convergence loop may produce units out of
    # original reading order. Audio playback requires units in
    # textual sequence. Stable-sort by character position ensures
    # correct TTS concatenation order.
    #
    # Contract: Units are substrings of sent_text. Position lookup
    # is O(n) per unit but total units are bounded by safety gates.
    # ────────────────────────────────────────────────────────────
    if final_units and len(final_units) > 1:
        def _unit_position(unit_text):
            """Return character index of unit in original sentence."""
            if not unit_text:
                return len(sent_text)
            # Use prefix for matching (handles minor whitespace variance)
            search_key = unit_text.strip()[:25]
            pos = sent_text.find(search_key)
            # Units not found (should not happen) sort to end
            return pos if pos >= 0 else len(sent_text)

        final_units.sort(key=_unit_position)

    return final_units if final_units else [sent_text.strip()]


def _soft_split_unit(
        text: str,
        trace_id: str = None,
        *,
        emergency: bool = False,
) -> List[str]:
    """
    TTS Unit Splitter — Tiered Strategy Cascade with Viability Enforcement.

    Professional narration pipeline model: split text into TTS-safe units
    using a strategy cascade, then enforce output viability so no unit is
    returned that would cause decoder runaway.

    Strategy Cascade (highest to lowest prosodic quality):
      L1  Delimiter boundaries — comma/semicolon with conjunction awareness.
          Produces the most natural clause-level units.
      L2  Balanced word-boundary midpoint — splits at the whitespace nearest
          to the character midpoint. Produces two roughly equal halves.

    Viability Gate (runs on ALL exits, not just emergency):
      Every output list passes through viability enforcement before return.
      Non-viable fragments (isolated compounds, orphaned notation) are merged
      with neighbors or decomposed into individual words. This eliminates
      "unsplittable; aborting chunk" failures by catching non-viable units
      during initial splitting — before they reach TTS.

    Entry Guard:
      Uses word-based threshold from extraction_engine._TTS_PROSODIC_MIN_CLAUSE_WORDS.
      Units below (MIN_CLAUSE_WORDS * 2) words without delimiter punctuation
      skip structural splitting (non-emergency only). Viability enforcement
      still runs on the early return.

    Args:
        text: Unit text to split.
        trace_id: Optional trace ID for observability logging.
        emergency: If True, skip entry guard (post-validation escalation).

    Returns:
        List of TTS-viable unit strings. Never empty if input is non-empty.
    """
    if not text:
        return []

    text = text.strip()
    word_count = len(text.split())

    min_words_for_split = int(
        getattr(extraction_engine, "_TTS_PROSODIC_MIN_CLAUSE_WORDS", 4)
    ) * 2

    # ════════════════════════════════════════════════════════════════
    # INLINE VIABILITY ENGINE
    #
    # Nested within _soft_split_unit to prevent module-level sprawl.
    # All viability logic is scoped to this method's responsibility:
    # ensuring every returned unit can be synthesized without decoder
    # runaway.
    # ════════════════════════════════════════════════════════════════

    def _is_viable(unit_text: str) -> bool:
        """
        Per-unit TTS pre-flight check.

        Viability model (empirically derived from TTS failure corpus):
          - 3+ words: sufficient prosodic context, always viable.
          - 2 words: adequate attention anchoring, viable.
          - 1 word with terminal punctuation: punct provides decoder
            anchor (comma, period, etc.), viable.
          - 1 word, simple alphabetic, no hyphens: common dictionary
            word, TTS handles reliably.
          - 1 word, contains digit: speakable numeric content
            (citations, statutory refs like "5 U.S.C."), viable.
          - 1 word, hyphenated compound, no punctuation: NOT VIABLE.
            Decoder treats hyphens as token boundaries, producing
            sub-tokens with no sentence-level context → runaway.
        """
        stripped = unit_text.strip()
        if not stripped:
            return False

        unit_word_count = len(stripped.split())

        # Multi-word units: always viable
        if unit_word_count >= _TTS_UNIT_MIN_VIABLE_WORDS:
            return True
        if unit_word_count == 2:
            return True

        # ── Single-word analysis ──
        has_prosodic_anchor = stripped[-1] in '.,;:!?'
        core = stripped.rstrip('.,;:!?"\')]}\u2019\u201d')

        # Hyphenated compound without anchor: HIGH RISK
        # "eighteen-year-olds" → 3 sub-tokens, no context → runaway
        # "eighteen-year-olds," → trailing comma anchors → viable
        if '-' in core:
            return has_prosodic_anchor

        # Simple alphabetic: always viable regardless of length
        # "Nevertheless", "Subsequently", "Geneva" — all succeed
        if core.isalpha():
            return True

        # Digit-containing: speakable numeric/citation content
        # "5 U.S.C.", "[12]", "§ 1983", "42 CFR" — TTS handles numbers
        # reliably. Prevents over-merging in legal/congressional texts.
        if any(c.isdigit() for c in core):
            return True

        # Remaining mixed tokens: require prosodic anchor
        return has_prosodic_anchor

    def _viability_gate(units: List[str]) -> List[str]:
        """
        Post-split viability enforcement.

        Runs on EVERY exit path. Ensures no unit is returned that would
        cause decoder runaway. Two repair strategies, ordered by audio
        quality:

          1. Merge non-viable fragment with nearest neighbor (multi-unit).
             Reconstructs natural phrase, preserves prosody. Prefer
             merge-left (preceding context sounds more natural).
          2. Decompose hyphenated compound into individual words
             (single-unit, no neighbor available). Last resort — each
             word is a common dictionary entry the model handles.

        Bounded at 3 passes to prevent pathological merge cascades.
        """
        if not units:
            return units

        result = list(units)

        for _ in range(3):
            changed = False
            i = 0

            while i < len(result):
                if _is_viable(result[i]):
                    i += 1
                    continue

                # ── Strategy 1: Merge with neighbor ──
                # Multi-unit list: merge produces a medium-sized unit
                # with natural phrase-level prosody. Strictly safer
                # than sending a fragment alone.
                if len(result) > 1:
                    if i > 0:
                        # Merge left (preferred: preceding context)
                        merged = result[i - 1].rstrip() + ' ' + result[i].lstrip()
                        if trace_id:
                            logger.debug(
                                "[%s] Viability merge-left: '%s' + '%s'",
                                trace_id,
                                result[i - 1][:30], result[i][:30]
                            )
                        result[i - 1] = merged
                        result.pop(i)
                        changed = True
                        continue
                    else:
                        # Merge right (first unit, no left neighbor)
                        merged = result[i].rstrip() + ' ' + result[i + 1].lstrip()
                        if trace_id:
                            logger.debug(
                                "[%s] Viability merge-right: '%s' + '%s'",
                                trace_id,
                                result[i][:30], result[i + 1][:30]
                            )
                        result[i] = merged
                        result.pop(i + 1)
                        changed = True
                        continue

                # ── Strategy 2: Hyphenated compound decomposition ──
                # Single unit, no neighbor to merge with. Decompose
                # at hyphens if all segments are real words (≥3 chars,
                # alphabetic). Preserves trailing punctuation on final
                # segment for decoder anchoring.
                stripped_unit = result[i].strip()
                if '-' in stripped_unit:
                    trailing_punct = ''
                    decomp_text = stripped_unit
                    if decomp_text and decomp_text[-1] in '.,;:!?':
                        trailing_punct = decomp_text[-1]
                        decomp_text = decomp_text[:-1]

                    segments = [s.strip() for s in decomp_text.split('-') if s.strip()]

                    if (len(segments) >= 2 and all(
                            len(s) >= _TTS_COMPOUND_SEGMENT_MIN_CHARS
                            and s.isalpha()
                            for s in segments
                    )):
                        if trailing_punct:
                            segments[-1] = segments[-1] + trailing_punct

                        if trace_id:
                            logger.debug(
                                "[%s] Compound decomposition: '%s' → %s",
                                trace_id, stripped_unit[:50], segments
                            )

                        result[i:i + 1] = segments
                        changed = True
                        i += len(segments)
                        continue

                # Non-viable but cannot be improved further.
                # Return as-is; TTS may still succeed, and the
                # escalation retry path provides a second chance.
                i += 1

            if not changed:
                break

        return result if result else units

    # ════════════════════════════════════════════════════════════════
    # ENTRY GUARD (non-emergency only)
    #
    # Small units without delimiter punctuation cannot be meaningfully
    # split at clause boundaries. Skip structural splitting but still
    # enforce viability — catches isolated compounds on the happy path
    # before they ever reach TTS.
    # ════════════════════════════════════════════════════════════════
    if not emergency:
        if word_count < min_words_for_split and (',' not in text and ';' not in text):
            return _viability_gate([text])

    # ════════════════════════════════════════════════════════════════
    # STRATEGY L1: Delimiter Boundaries (comma / semicolon)
    #
    # Split on punctuation delimiters with conjunction-aware flushing.
    # Conjunctions (and, or, but, nor, yet) bind to the following
    # clause rather than stranding at the end of the current unit.
    # ════════════════════════════════════════════════════════════════
    parts = re.split(r'([,;]\s*)', text)

    if len(parts) <= 1:
        # ────────────────────────────────────────────────────────
        # STRATEGY L2: Balanced Word-Boundary Midpoint
        #
        # No delimiter boundaries exist. Split at the whitespace
        # nearest to the character midpoint for two roughly balanced
        # halves. Viability gate on the result handles any fragment
        # that is too small or isolated.
        # ────────────────────────────────────────────────────────
        text_len = len(text)
        midpoint = text_len // 2

        left = text.rfind(' ', 0, midpoint)
        right = text.find(' ', midpoint)

        if left == -1 and right == -1:
            # No whitespace (single token). Viability gate will
            # decompose if compound, or return as-is.
            return _viability_gate([text])

        if left == -1:
            split_at = right
        elif right == -1:
            split_at = left
        else:
            split_at = (
                left if (midpoint - left) <= (right - midpoint) else right
            )

        # Guard: avoid splitting inside brackets, parens, or
        # citations like [17] or (b)(4). Nudge split point left
        # to the preceding whitespace if we'd land after an opener.
        _safe_split = split_at
        while _safe_split > 0 and text[_safe_split - 1] in '([{':
            _safe_split = text.rfind(' ', 0, _safe_split - 1)
            if _safe_split == -1:
                _safe_split = split_at
                break

        a = text[:_safe_split].strip()
        b = text[_safe_split:].strip()
        if a and b:
            return _viability_gate([a, b])

        return _viability_gate([text])

    # ────────────────────────────────────────────────────────────
    # L1 Reconstruction: Flush at delimiter boundaries
    #
    # Walk the parts list, accumulating content into a buffer.
    # On each delimiter, decide whether to flush (create a new unit)
    # or continue (conjunction lookahead keeps phrase together).
    # ────────────────────────────────────────────────────────────
    units = []
    buffer = ""
    idx = 0

    while idx < len(parts):
        seg = parts[idx] or ""
        if not seg:
            idx += 1
            continue

        is_delim = bool(re.match(r'^[,;]\s*$', seg))

        if is_delim:
            buffer += seg

            # Lookahead: does next content start with a conjunction?
            next_content = ""
            if idx + 1 < len(parts):
                next_content = (parts[idx + 1] or "").strip().lower()

            # Flush unless next segment opens with a conjunction
            if buffer.strip() and not next_content.startswith(
                    ('and ', 'or ', 'but ', 'nor ', 'yet ')
            ):
                u = buffer.strip()
                if u:
                    units.append(u)
                buffer = ""

            idx += 1
            continue

        # Content segment
        buffer += seg
        idx += 1

    # Finalize remaining buffer
    if buffer.strip():
        units.append(buffer.strip())

    # ────────────────────────────────────────────────────────────
    # Filter and validate
    # ────────────────────────────────────────────────────────────
    units = [u for u in units if u and u.strip()]

    if not units:
        return _viability_gate([text])

        # ────────────────────────────────────────────────────────────
        # L1→L2 FALLTHROUGH: Conjunction Merge Recovery
        # L1 can legally produce a single unit identical to the input
        # when the only comma is followed by a conjunction ("period,
        # and it decreased..."). Conjunction-aware flushing merges the
        # split back together — correct for prosody but leaves the
        # full-length unit intact for TTS, risking decoder drift.
        #
        # Fall through to L2 midpoint when clause structure is weak
        # AND the unit exceeds proactive thresholds. Same structural
        # definition as work queue has_strong_clause_structure:
        #   strong = (commas >= 2 OR semicolons >= 1)
        #   weak = NOT strong AND no colon
        # ────────────────────────────────────────────────────────────
    if len(units) == 1 and units[0].strip() == text.strip():
        _l1_strong_clause = (
                text.count(',') >= 2 or
                text.count(';') >= 1
        )
        _l1_weak_clause = (
                not _l1_strong_clause and
                ':' not in text
        )
        if (
                _l1_weak_clause and
                len(text) > _TTS_PROACTIVE_SPLIT_CHARS and
                word_count > _TTS_PROACTIVE_SPLIT_WORDS
        ):
            # L1→L2 fallthrough: conjunction merge defeated L1
            _ft_len = len(text)
            _ft_mid = _ft_len // 2
            _ft_left = text.rfind(' ', 0, _ft_mid)
            _ft_right = text.find(' ', _ft_mid)

            if _ft_left != -1 or _ft_right != -1:
                if _ft_left == -1:
                    _ft_split = _ft_right
                elif _ft_right == -1:
                    _ft_split = _ft_left
                else:
                    _ft_split = (
                        _ft_left if (_ft_mid - _ft_left) <= (_ft_right - _ft_mid)
                        else _ft_right
                    )

                _ft_safe = _ft_split
                while _ft_safe > 0 and text[_ft_safe - 1] in '([{':
                    _ft_safe = text.rfind(' ', 0, _ft_safe - 1)
                    if _ft_safe == -1:
                        _ft_safe = _ft_split
                        break

                _ft_a = text[:_ft_safe].strip()
                _ft_b = text[_ft_safe:].strip()
                if _ft_a and _ft_b:
                    if trace_id:
                        logger.debug(
                            "[%s] L1→L2 fallthrough: conjunction merge "
                            "defeated L1, midpoint split %d/%d chars",
                            trace_id, len(_ft_a), len(_ft_b)
                        )
                    return _viability_gate([_ft_a, _ft_b])

        return _viability_gate([text])

    # ════════════════════════════════════════════════════════════════
    # VIABILITY GATE — Final enforcement on all L1 split results.
    # Non-viable fragments merged with neighbors or decomposed
    # before return. Caller never receives a unit likely to cause
    # decoder runaway.
    # ════════════════════════════════════════════════════════════════
    return _viability_gate(units)


def _validate_tts_audio(
        audio_bytes: bytes,
        expected_duration: float,
        trace_id: str,
        chunk_id: int,
) -> bool:
    """
    Validate TTS output duration against expected.
    Returns False if audio is grossly inflated/deflated (junk).

    Gate order:
        1. Parse WAV → extract actual duration
        2. Enforce absolute ceiling (when WAV readable)
        3. Fail-open if expected is None/non-numeric/tiny
        4. Enforce ratio bounds
    """
    # ════════════════════════════════════════════════════════════════
    # STEP 1: Parse WAV to extract actual duration
    # Must happen first so ceiling can be enforced regardless of expected
    # ════════════════════════════════════════════════════════════════
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            actual = w.getnframes() / float(w.getframerate() or 1)
    except Exception as e:
        logger.warning(f"[{trace_id}] Chunk {chunk_id} WAV validation failed: {e}")
        return True  # Fail-open on parse error (ceiling cannot be enforced)

    # ════════════════════════════════════════════════════════════════
    # STEP 2: Absolute ceiling — ALWAYS enforced when WAV is readable
    # ════════════════════════════════════════════════════════════════
    if actual > _TTS_MAX_WAV_SECONDS:
        logger.error(
            f"[{trace_id}] Chunk {chunk_id} rejected: actual={actual:.1f}s exceeds {_TTS_MAX_WAV_SECONDS}s"
        )
        return False

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Fail-open for missing/non-numeric/tiny expected duration
    # Ratio checks not meaningful without valid expected baseline
    # ════════════════════════════════════════════════════════════════
    if expected_duration is None:
        return True  # No expectation available; fail-open on ratio

    try:
        expected_duration = float(expected_duration)
    except Exception:
        return True  # Non-numeric; fail-open on ratio

    if expected_duration < _TTS_MIN_EXPECTED_SECONDS:
        return True  # Tiny expectation; ratio checks not meaningful

    # ════════════════════════════════════════════════════════════════
    # STEP 4: Ratio-based validation
    # ════════════════════════════════════════════════════════════════
    ratio = actual / float(expected_duration)

    if ratio > _TTS_MAX_DURATION_RATIO:
        logger.error(
            f"[{trace_id}] Chunk {chunk_id} rejected: ratio={ratio:.1f}x exceeds {_TTS_MAX_DURATION_RATIO}x"
        )
        return False

    if ratio < _TTS_MIN_DURATION_RATIO:
        logger.error(
            f"[{trace_id}] Chunk {chunk_id} rejected: ratio={ratio:.2f}x below {_TTS_MIN_DURATION_RATIO}x"
        )
        return False

    return True


async def generate_single_chunk(
        chunk: dict,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        _logger
):
    """
    Returns:
        bool: True if chunk audio is ready or successfully generated,
              False on any recoverable failure.
    """
    chunk_id = chunk['chunk_id']
    page = chunk['page']
    audio_filename = f"chunk_{chunk_id:04d}_p{page}.wav"
    audio_path = OUTPUT_DIR / book_id / audio_filename
    ids = _identity_from_manifest(book_id, trace_id)
    logger = _logger

    async with TTS_SEMAPHORE:
        # ────────────────────────────────────────────────────────────────
        # RECOVERY PATH: If audio already exists, register in manifest
        # ────────────────────────────────────────────────────────────────
        if audio_path.exists():
            async with MANIFEST_LOCK:
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    if not any(c.get('chunk_id') == chunk_id
                               for c in manifest.get('ready_chunks', [])):
                        audio_ref = _register_file_artifact(
                            path=audio_path,
                            role="audio_chunk",
                            trace_id=trace_id,
                            book_id=book_id,
                            document_id=ids.get("document_id"),
                            job_id=ids.get("job_id"),
                            audiobook_id=ids.get("audiobook_id"),
                            mime_type="audio/wav",
                            schema_version="tts.audio_chunk.v1",
                            metadata={"chunk_id": chunk_id, "page": page,
                                      "recovered_from_disk": True},
                        )
                        manifest['ready_chunks'].append({
                            "chunk_id": chunk_id,
                            "filename": audio_filename,
                            "page": page,
                            "pages": (lambda _c, _p: (lambda _s: sorted(
                                set([x for x in (_c.get('pages') or []) if x is not None] + (
                                    [_p] if _p is not None else []) + [y for y in _s if
                                                                       y is not None])))(
                                [(ss.get('page_number') if isinstance(ss, dict) else None) for ss in
                                 (_c.get('sentences') or [])]))(chunk, page),
                            "text_snippet": chunk['text'][:50] + "...",
                            "start_time": chunk['start_time'],
                            "duration_seconds": chunk['duration_seconds'],
                            "end_time": chunk['end_time'],
                            "sentences": chunk.get('sentences', []),
                            "artifact_ref": audio_ref,
                        })
                        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])
                        _merge_manifest_artifact_refs(manifest, [audio_ref])
                        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                        logger.info(f"[{trace_id}] Chunk {chunk_id} recovered from disk.")
                    return True
                except Exception as e:
                    logger.error(
                        f"[{trace_id}] Failed to recover chunk {chunk_id}: {e}",
                        exc_info=True
                    )
                    return False

        # ====================================================================
        # NOTE (V1.8): Chunk-level overflow guard removed.
        # TTS is invoked per-sentence/clause only.
        # Safety is enforced structurally by synthesis granularity.
        # Per-sentence guard added below in synthesis loop.
        # ====================================================================
        # Generate new audio
        try:
            # ================================================================
            # V1.8: Per-Sentence TTS Generation (Always)
            # ================================================================
            # Every sentence is synthesized independently. If any sentence
            # fails, the entire chunk fails (preserves Stage-2 timing authority).
            #
            # PATH LOGIC (UNIFIED):
            #   For each sentence:
            #     → Unitize via _prepare_synthesis_units (uses Stage 2 hints)
            #     → Synthesize each unit individually
            #     → Concatenate units into sentence audio
            #   Concatenate all sentence audios into chunk audio.
            #
            # TIMING CONTRACT:
            #   Chunk fails if any sentence fails. This ensures
            #   chunk["duration_seconds"] and per-sentence timing fields
            #   remain authoritative. No silent drift.
            # ================================================================
            sentences = chunk.get('sentences', [])
            if not sentences:
                logger.warning(f"[{trace_id}] Chunk {chunk_id} has no sentences")
                return False

            logger.info(
                f"[{trace_id}] Chunk {chunk_id} START: "
                f"{len(sentences)} sentences, {len(chunk.get('text', ''))} chars"
            )

            # ════════════════════════════════════════════════════════════════
            # PER-SENTENCE SYNTHESIS (always this path)
            # ════════════════════════════════════════════════════════════════
            audio_segments = []
            sentences_succeeded = 0

            for sent_idx, sent in enumerate(sentences):
                # Contract guard: sentence must be dict
                if not isinstance(sent, dict):
                    logger.error(
                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} is not a dict; "
                        f"failing chunk to preserve contract integrity"
                    )
                    return False

                sent_text = sent.get('text', '')
                sent_text = sent_text.strip() if isinstance(sent_text, str) else ""
                if not sent_text:
                    continue

                # ── AUTHORITATIVE SENTENCE TIMING CONTEXT ──
                sentence_expected_duration = sent.get("duration_seconds", 0.0)
                sentence_text_len = len(sent_text)

                # ────────────────────────────────────────────────────────────
                # UNIFIED PATH: Always unitize sentence before TTS
                # ────────────────────────────────────────────────────────────
                synthesis_units = _prepare_synthesis_units(
                    sent,
                    trace_id=trace_id,
                    chunk_id=chunk_id,
                    sent_idx=sent_idx,
                ) or []

                synthesis_units = [u.strip() for u in synthesis_units if u and u.strip()]
                if not synthesis_units:
                    logger.error(
                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx}: "
                        f"no synthesis units produced; failing chunk to preserve timing"
                    )
                    return False

                unit_segments = []

                # ════════════════════════════════════════════════════════════
                # POST-VALIDATION ESCALATION
                # ════════════════════════════════════════════════════════════
                # Process units as a mutable queue. If a unit fails validation
                # after retries, split it further and retry the sub-units.
                # This closes the loop: pre-TTS splitting → TTS → validation → escalation
                # ════════════════════════════════════════════════════════════
                unit_idx = 0
                escalation_count = 0

                while unit_idx < len(synthesis_units):
                    unit = synthesis_units[unit_idx]
                    unit = unit.strip() if isinstance(unit, str) else ""
                    if not unit:
                        unit_idx += 1
                        continue

                    unit_audio = None
                    unit_params = {
                        "text": unit,
                        "speaker_id": "",
                        "style_wav": "",
                        "language_id": ""
                    }

                    # ── UNIT-LEVEL EXPECTED DURATION (DERIVED) ──
                    # Floor prevents ratio explosion for short units where TTS has
                    # irreducible minimum output (~1.2-1.5s for any token).
                    if sentence_expected_duration and sentence_text_len > 0:
                        unit_expected_duration = max(
                            sentence_expected_duration * (len(unit) / sentence_text_len),
                            _TTS_MIN_EXPECTED_SECONDS,
                        )
                    else:
                        unit_expected_duration = max(
                            sentence_expected_duration or 0.0,
                            _TTS_MIN_EXPECTED_SECONDS,
                        )

                    for _attempt in range(2):
                        try:
                            r = await client.post(
                                TTS_SERVICE_URL, data=unit_params, timeout=180.0
                            )
                            r.raise_for_status()
                            candidate_audio = r.content

                            if not _validate_tts_audio(
                                    candidate_audio,
                                    unit_expected_duration,
                                    trace_id,
                                    chunk_id,
                            ):
                                if _attempt == 0:
                                    logger.warning(
                                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "
                                        f"unit {unit_idx} failed validation, retrying..."
                                    )
                                    continue

                                sub_units = _soft_split_unit(unit, trace_id, emergency=True) or []
                                sub_units = [x.strip() for x in sub_units if x and x.strip()]

                                if len(sub_units) <= 1 or (
                                        len(sub_units) == 1 and sub_units[0] == unit):
                                    logger.error(
                                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "
                                        f"unit {unit_idx} failed validation and is unsplittable; "
                                        f"aborting chunk. text='{unit[:80]}...'"
                                    )
                                    return False

                                escalation_count += 1
                                logger.info(
                                    f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "
                                    f"unit {unit_idx} ESCALATING (depth={escalation_count}): "
                                    f"split into {len(sub_units)} sub-units"
                                )

                                # Replace this unit with sub-units in-place
                                synthesis_units[unit_idx:unit_idx + 1] = sub_units
                                unit_audio = None
                                break

                            unit_audio = candidate_audio
                            break


                        except (httpx.TimeoutException, httpx.NetworkError,
                                httpx.HTTPStatusError) as e:
                            if _attempt == 0:
                                logger.warning(

                                    f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} "

                                    f"unit {unit_idx} transient TTS error ({type(e).__name__}), retrying..."

                                )
                                continue
                            raise

                    # If escalation happened, continue at same index (now pointing to first sub-unit)
                    if unit_audio is None:
                        continue

                    unit_segments.append(unit_audio)
                    unit_idx += 1

                # Concatenate units into sentence audio
                sentence_audio = _concatenate_wav_with_gaps(unit_segments, gap_ms=40)

                # ────────────────────────────────────────────────────────────
                # TIMING CONTRACT: Fail chunk if sentence fails
                # ────────────────────────────────────────────────────────────
                if not sentence_audio:
                    logger.error(
                        f"[{trace_id}] Chunk {chunk_id} sent {sent_idx} produced no audio; "
                        f"failing chunk to preserve timing"
                    )
                    return False

                audio_segments.append(sentence_audio)
                sentences_succeeded += 1

            # ════════════════════════════════════════════════════════════════
            # CONCATENATE SENTENCES INTO CHUNK AUDIO
            # ════════════════════════════════════════════════════════════════
            if not audio_segments:
                logger.error(f"[{trace_id}] Chunk {chunk_id} produced no audio segments")
                return False

            final_audio = _concatenate_wav_with_gaps(audio_segments, gap_ms=40)

            # Compute expected duration from Stage 2 sentence timing
            # NOTE: Only count sentences that were actually synthesized
            # (empty/whitespace sentences were skipped in the loop above)
            expected_duration = 0.0
            duration_count = 0
            for _s in sentences:
                _s_text = _s.get('text', '')
                if not _s_text or not _s_text.strip():
                    continue  # Skip empty sentences (matches synthesis loop)
                ds = _s.get("duration_seconds")
                if isinstance(ds, (int, float)):
                    expected_duration += float(ds)
                    duration_count += 1

            if duration_count == 0:
                logger.error(
                    f"[{trace_id}] Chunk {chunk_id} has no valid sentence durations; "
                    f"failing chunk to avoid timing drift"
                )
                return False

            if not _validate_tts_audio(
                    final_audio,
                    expected_duration,
                    trace_id,
                    chunk_id,
            ):
                return False

            # Write audio file atomically so restart/retry never sees a half-written chunk.
            write_bytes_atomic(audio_path, final_audio)

            # Verify write succeeded (defense against partial writes)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                logger.error(
                    f"[{trace_id}] Chunk {chunk_id} audio write failed or empty"
                )
                if audio_path.exists():
                    audio_path.unlink()
                return False

            logger.info(
                f"[{trace_id}] Chunk {chunk_id} generated via sentence-first path "
                f"({sentences_succeeded}/{len(sentences)} sentences)"
            )

            # Register in manifest
            async with MANIFEST_LOCK:
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if not any(c.get('chunk_id') == chunk_id for c in
                               manifest.get('ready_chunks', [])):
                        audio_ref = _register_file_artifact(
                            path=audio_path,
                            role="audio_chunk",
                            trace_id=trace_id,
                            book_id=book_id,
                            document_id=ids.get("document_id"),
                            job_id=ids.get("job_id"),
                            audiobook_id=ids.get("audiobook_id"),
                            mime_type="audio/wav",
                            schema_version="tts.audio_chunk.v1",
                            metadata={"chunk_id": chunk_id, "page": page},
                        )
                        manifest['ready_chunks'].append({
                            "chunk_id": chunk_id,
                            "filename": audio_filename,
                            "page": page,
                            "pages": chunk.get('pages') or (
                                [page] if page is not None else []),
                            "text_snippet": chunk['text'][:50] + "...",
                            "start_time": chunk['start_time'],
                            "duration_seconds": chunk['duration_seconds'],
                            "end_time": chunk['end_time'],
                            "sentences": chunk.get('sentences', []),
                            "artifact_ref": audio_ref,
                        })
                        manifest['ready_chunks'].sort(key=lambda c: c['chunk_id'])
                        _merge_manifest_artifact_refs(manifest, [audio_ref])
                        validate_and_write_manifest(manifest_path, manifest, trace_id, logger)
                except Exception as e:
                    logger.error(
                        f"[{trace_id}] Failed to update manifest for chunk {chunk_id}: {e}"
                    )

        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as e:
            logger.error(
                f"[{trace_id}] TTS HTTP failure for chunk {chunk_id}: {e}",
                exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"[{trace_id}] TTS internal error for chunk {chunk_id}: {e}",
                exc_info=True
            )
            return False

        return True

    # ------------------------------------------------------------------
    # SAFETY NET:
    # This function is semantically boolean. Falling through without an
    # explicit return indicates a logic regression.
    # ------------------------------------------------------------------
    logger.error(
        f"[{trace_id}] INTERNAL ERROR: generate_single_chunk() "
        f"fell through without return (chunk_id={chunk_id})"
    )
    return False


async def generate_audio_streaming(
        citation_json_path: Path,
        book_id: str,
        trace_id: str,
        manifest_path: Path,
        limit=None,
        target_chunk_ids: Optional[Set[int]] = None
):
    if not citation_json_path.exists():
        return False

    with open(citation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    (OUTPUT_DIR / book_id).mkdir(parents=True, exist_ok=True)

    chunks = data['chunks'][:limit] if limit else data['chunks']

    # Selective rebuild filter: generate only targeted chunks
    if target_chunk_ids is not None:
        pre_filter_count = len(chunks)
        chunks = [c for c in chunks if c.get('chunk_id') in target_chunk_ids]
        logger.info(
            f"[{trace_id}] Stage 3 selective: "
            f"{len(chunks)}/{pre_filter_count} chunks targeted"
        )

    if not chunks:
        logger.warning(f"[{trace_id}] Stage 3 invoked with zero chunks")
        return False

    _record_job_stage(
        manifest_path, trace_id,
        status="running",
        stage="stage_3_audio",
        message=(
            f"Stage 3 queued {len(chunks)} audio chunks with "
            f"{min(TTS_STAGE3_WORKER_COUNT, len(chunks))} bounded workers."
        ),
        event_type="stage_3_chunks_queued",
        progress_total=len(chunks),
        metadata={
            "worker_count": min(TTS_STAGE3_WORKER_COUNT, len(chunks)),
            "execution_policy": "bounded_worker_queue",
        },
    )

    try:
        result = await _generate_audio_chunks_bounded(
            chunks=chunks,
            book_id=book_id,
            trace_id=trace_id,
            manifest_path=manifest_path,
        )

        successes = int(result.get("successes") or 0)
        failures = int(result.get("failures") or 0)
        failed_chunks = result.get("failed_chunks") or []

        for failure in failed_chunks:
            logger.error(
                f"[{trace_id}] Chunk {failure.get('chunk_id', '?')} failed during Stage 3: "
                f"{failure.get('reason', 'generation_failed')}"
            )

        logger.info(
            f"[{trace_id}] Stage 3 Summary: {successes}/{len(chunks)} chunks generated "
            f"using {result.get('worker_count')} bounded workers"
        )

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            m["stage3_worker_count"] = result.get("worker_count")
            m["chunk_state_dir"] = str(_stage3_chunk_checkpoint_dir(book_id))
            if failures > 0:
                logger.warning(f"[{trace_id}] {failures} chunks failed to generate")
                m["processing_status"] = "stage_3_partial"
                m["error_message"] = f"{failures} chunks failed during Stage 3"
                m["failed_chunks"] = failed_chunks
                m["stage3_failed_chunks"] = failed_chunks
            else:
                m["failed_chunks"] = []
                m["stage3_failed_chunks"] = []
            validate_and_write_manifest(manifest_path, m, trace_id, logger)
        except Exception as e:
            logger.error(
                f"[{trace_id}] Failed to update manifest after Stage 3 summary: {e}"
            )

        try:
            _write_audio_timing_artifact(
                book_id=book_id,
                trace_id=trace_id,
                manifest_path=manifest_path,
                citation_json_path=citation_json_path,
            )
        except Exception as exc:
            logger.warning("[%s] Failed to generate audio timing artifact: %s", trace_id, exc)

        return successes == len(chunks)
    except Exception as e:
        logger.error(f"[{trace_id}] Stage 3 Crash: {e}", exc_info=True)
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
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
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        audio_dir = OUTPUT_DIR / book_id
        actual_files = {f.name for f in audio_dir.glob("chunk_*.wav")}
        manifest_files = {
            c.get('filename')
            for c in manifest.get('ready_chunks', [])
            if c.get('filename')
        }

        # Remove entries for missing files
        orphaned = manifest_files - actual_files
        if orphaned:
            manifest['ready_chunks'] = [
                c for c in manifest['ready_chunks'] if c['filename'] not in orphaned
            ]
            logger.warning(f"[{trace_id}] Removed {len(orphaned)} orphaned manifest entries")

        # Log any files missing from manifest. Do not invent ready_chunks without
        # semantic/timing data; register them as orphaned artifacts for inspection.
        missing = actual_files - manifest_files
        if (
                missing
                and manifest.get("processing_status") in ("stage_3_started", "stage_3_partial")
                and manifest.get("total_chunks", 0) > 0
        ):
            logger.warning(
                f"[{trace_id}] {len(missing)} audio files not in manifest; "
                "registering as orphaned audio artifacts without ready_chunk recovery"
            )

            orphaned_audio = manifest.get("orphaned_audio_artifacts", [])
            orphan_refs = []
            ids = _identity_from_manifest(book_id, trace_id)
            for filename in sorted(missing):
                audio_path = audio_dir / filename
                if not audio_path.exists():
                    continue
                match = re.match(r'chunk_(\d+)_p(\d+)\.wav', filename)
                metadata = {"filename": filename, "orphaned": True}
                if match:
                    metadata["chunk_id"] = int(match.group(1))
                    metadata["page"] = int(match.group(2))
                audio_ref = _register_file_artifact(
                    path=audio_path,
                    role="orphaned_audio_chunk",
                    trace_id=trace_id,
                    book_id=book_id,
                    document_id=ids.get("document_id"),
                    job_id=ids.get("job_id"),
                    audiobook_id=ids.get("audiobook_id"),
                    mime_type="audio/wav",
                    schema_version="tts.audio_chunk.v1",
                    metadata=metadata,
                )
                orphan_refs.append(audio_ref)
                if not any(item.get("artifact_id") == audio_ref.get("artifact_id") for item in
                           orphaned_audio):
                    orphaned_audio.append(audio_ref)

            if orphan_refs:
                manifest["orphaned_audio_artifacts"] = orphaned_audio
                _merge_manifest_artifact_refs(manifest, orphan_refs)

        elif missing:
            logger.warning(f"[{trace_id}] {len(missing)} audio files not in manifest: {missing}")

        # Update status based on reconciled count
        completed = len(manifest['ready_chunks'])
        total = manifest['total_chunks']

        if total > 0 and completed == total:
            manifest['processing_status'] = 'stage_3_complete'
        elif completed > 0:
            if manifest.get('processing_status') != 'stage_3_complete':
                manifest['processing_status'] = 'stage_3_partial'
        else:
            manifest['processing_status'] = 'stage_3_failed'

        # Build sentence_index for O(1) frontend lookup
        sentence_index = {}
        for chunk in manifest.get('ready_chunks', []):
            chunk_id = chunk.get('chunk_id')
            for sent in chunk.get('sentences', []):
                sid = sent.get('global_index')
                if sid is not None:
                    sentence_index[sid] = {
                        'chunk_id': chunk_id,
                        **sent
                    }
        manifest['sentence_index'] = sentence_index

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
        manifest_path = Path(path)
        payload = _with_job_progress_fields(dict(data or {}))
        payload.setdefault("created_at", utc_now_iso())
        payload["updated_at"] = utc_now_iso()
        validated = ManifestSchema(**payload)
        output = _manifest_dump(validated)
        atomic_write_manifest(manifest_path, output, _logger)
        if manifest_path.name == "manifest.json":
            _register_manifest_artifact(manifest_path, output, trace_id)
            _sync_manifest_job_state(manifest_path, output)
    except Exception as e:
        logger.error(f"Manifest Error: {e}")
        raise e


def get_citation_at_timestamp(path, timestamp):
    """
    API implementation for citation lookup.

    Uses authoritative per-sentence timing and filters excluded spans from
    source_cids for UI highlighting. Chunks without sentence timing are not
    treated as highlightable.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # ═══════════════════════════════════════════════════════════════════
        # LOAD SEMANTIC LOOKUP FOR EXCLUSION FILTERING (fail-open)
        # ═══════════════════════════════════════════════════════════════════
        span_lookup = {}
        try:
            semantic_path = Path(path).with_name(
                Path(path).name.replace("_citation_ready.json", "_semantic.json")
            )
            if semantic_path.exists():
                with open(semantic_path, 'r', encoding='utf-8') as sf:
                    span_lookup = json.load(sf).get("spans", {})
        except Exception:
            span_lookup = {}  # Fail-open: no filtering if semantic unavailable

        for chunk in data.get('chunks', []):
            # Check if timestamp falls within this chunk
            if chunk['start_time'] <= timestamp < chunk['end_time'] + 0.0001:
                sentences = chunk.get('sentences', [])
                if not sentences:
                    continue

                # Defensive ordering: never trust implicit sentence order
                sentences = sorted(
                    sentences,
                    key=lambda s: (
                        s.get('start_time', float('inf')),
                        s.get('global_index', float('inf'))
                    )
                )

                for sent in sentences:
                    sent_start = sent.get('start_time')
                    sent_end = sent.get('end_time')
                    if sent_start is None or sent_end is None:
                        continue
                    if sent_start <= timestamp < sent_end + 0.0001:
                        source_cids = (
                                sent.get('source_cids')
                                or sent.get('_source_span_ids')
                                or sent.get('_source_cids')
                                or sent.get('source_span_ids')
                        )
                        if source_cids:
                            source_cids = [cid for cid in source_cids if cid is not None]
                        else:
                            source_cids = []

                        source_cids = _filter_excluded_cids(source_cids, span_lookup)

                        return {
                            'page': sent.get('page_number', chunk['page']),
                            'span_start_index': sent.get('span_start_index'),
                            'span_end_index': sent.get('span_end_index'),
                            'source_cids': source_cids,
                            'role': sent.get('role', 'body'),
                            'highlighting_enabled': data.get('highlighting_enabled', True),
                            'sentence_text': sent.get('text', '')[:50],
                        }

                return None


    except Exception as e:
        # Citation lookup must never crash the API,
        # but silent failure makes debugging impossible.
        try:
            logger.error(
                "[citation] Failed lookup at timestamp %.3f: %s",
                timestamp, str(e),
                exc_info=True
            )
        except Exception:
            pass

    return None