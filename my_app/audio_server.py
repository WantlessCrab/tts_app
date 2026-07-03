# ~/TTS/my_app/audio_server.py

# SECTION 0 — IMPORTS, CONFIGURATION, CONSTANTS
from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Optional

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .artifact_store import (
    find_artifact_by_id,
    find_artifact_by_role,
    load_audiobook_artifacts,
    load_document_asset,
    load_trace_catalog,
)
from .gateway_router import route_ingest
from .job_store import list_job_events, list_jobs, load_job, load_trace_jobs
from .service_health import build_health_response

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

# --- Container Service URLs ---
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://pdf-processor-service:8000")
ACQUIRE_SERVICE_URL = os.getenv("ACQUIRE_SERVICE_URL", "http://host.docker.internal:8005")
CONVERT_SERVICE_URL = os.getenv("CONVERT_SERVICE_URL", "http://host.docker.internal:8006")

app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

AUDIO_SOURCES = {
    "audiobooks": AUDIOBOOKS_DIR,  # /workspace/outputs/audiobooks
    "obsidian": OBSIDIAN_DIR,  # /workspace/obsidian_audio
    "standalone": OUTPUT_DIR  # /workspace/outputs (for non-audiobook files)
}
# Use lowercase default consistent with keys
DEFAULT_AUDIO_SOURCE_NAME = "audiobooks"
STALE_JOB_THRESHOLD_MINUTES = int(os.getenv('STALE_JOB_THRESHOLD_MINUTES', '30'))
ARTIFACT_SHARD_DIRNAME = "artifact_shards"
UI_VENDOR_ASSETS = {
    "pdfjs": STATIC_DIR / "vendor" / "pdfjs" / "pdf.min.js",
    "pdfjs_worker": STATIC_DIR / "vendor" / "pdfjs" / "pdf.worker.min.js",
    "wavesurfer": STATIC_DIR / "vendor" / "wavesurfer" / "wavesurfer.min.js",
}

# APPLICATION SETUP

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the HTTP client for calling other services
client = httpx.AsyncClient(timeout=30.0)


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=1)
    trace_id: Optional[str] = None


async def _downstream_health(name: str, base_url: str) -> str:
    try:
        response = await client.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        if response.status_code == 200:
            return "ok"
        return f"error: HTTP {response.status_code}"
    except Exception as exc:
        return f"error: {exc}"


def _safe_workspace_path(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    workspace = WORKSPACE_DIR.resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path is outside workspace") from exc
    return path


def _artifact_file_response(artifact: dict, *,
                            fallback_media_type: str = "application/octet-stream") -> FileResponse:
    path_value = artifact.get("path")
    if not path_value:
        raise HTTPException(status_code=404, detail="Artifact has no path")
    path = _safe_workspace_path(str(path_value))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(
        path=path,
        media_type=artifact.get("mime_type") or fallback_media_type,
        headers={
            "X-Artifact-ID": str(artifact.get("artifact_id") or ""),
            "X-Artifact-Role": str(artifact.get("role") or ""),
            "Cache-Control": "public, max-age=3600",
        },
    )


def _artifact_for_book_role(book_id: str, role: str) -> Optional[dict]:
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    return find_artifact_by_role(AUDIOBOOKS_DIR / safe_book_id, role)


def _safe_book_dirname(book_id: str) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^\w\s-]', '', book_id).strip().replace(' ', '_'))


def _load_shard_index(book_id: str, role: str) -> tuple[dict, Path, Optional[dict]]:
    safe_role = re.sub(r"[^\w\-]+", "_", role).strip("_")
    if safe_role not in {"semantic", "ui_sentences"}:
        raise HTTPException(status_code=400, detail="Unsupported shard role")

    index_artifact = _artifact_for_book_role(book_id, f"{safe_role}_shard_index")
    if index_artifact and index_artifact.get("path"):
        index_path = _safe_workspace_path(str(index_artifact["path"]))
    else:
        index_path = PDF_CACHE_DIR / ARTIFACT_SHARD_DIRNAME / _safe_book_dirname(
            book_id) / safe_role / "index.json"
        index_path = _safe_workspace_path(str(index_path))

    if not index_path.exists() or not index_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"{safe_role} shard index not found for audiobook '{book_id}'")

    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
    except Exception as exc:
        logger.error("Failed to read shard index %s: %s", index_path, exc)
        raise HTTPException(status_code=500, detail="Failed to read shard index") from exc

    if not isinstance(index_data, dict):
        raise HTTPException(status_code=500, detail="Shard index is not a JSON object")
    return index_data, index_path.parent, index_artifact


def _shard_index_response(book_id: str, role: str) -> FileResponse:
    _, root, artifact = _load_shard_index(book_id, role)
    index_path = root / "index.json"
    headers = {"Cache-Control": "public, max-age=3600"}
    if artifact:
        headers["X-Artifact-ID"] = str(artifact.get("artifact_id") or "")
        headers["X-Artifact-Role"] = str(artifact.get("role") or "")
    return FileResponse(path=index_path, media_type="application/json", headers=headers)


def _shard_page_response(book_id: str, role: str, page_number: int) -> FileResponse:
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")
    index_data, root, _ = _load_shard_index(book_id, role)
    page_entry = None
    for item in index_data.get("pages") or []:
        if int(item.get("page_number") or -1) == page_number:
            page_entry = item
            break
    if not page_entry:
        raise HTTPException(status_code=404, detail=f"No {role} shard found for page {page_number}")
    filename = re.sub(r"[^\w\-\.]+", "", str(page_entry.get("filename") or ""))
    if not filename:
        raise HTTPException(status_code=500, detail="Shard index entry has no filename")
    shard_path = _safe_workspace_path(str(root / filename))
    if not shard_path.exists() or not shard_path.is_file():
        raise HTTPException(status_code=404, detail=f"Shard file not found for page {page_number}")
    return FileResponse(
        path=shard_path,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


def _load_job_for_manifest(manifest: dict) -> Optional[dict]:
    job_id = manifest.get("job_id")
    if not job_id:
        return None
    return load_job(str(job_id), cache_dir=PDF_CACHE_DIR)


def _local_ui_dependency_status() -> dict:
    assets = {}
    for key, path in UI_VENDOR_ASSETS.items():
        assets[key] = {
            "path": str(path),
            "available": path.exists() and path.is_file(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
    return {
        "status": "ok" if all(item["available"] for item in assets.values()) else "degraded",
        "assets": assets,
        "policy": "local_static_assets_only",
    }


def _safe_audio_source_file(source: str, filename: str, book_id: Optional[str] = None) -> Path:
    if source not in AUDIO_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio source specified. Valid sources: {list(AUDIO_SOURCES.keys())}",
        )

    safe_filename = re.sub(r"[^\w\-\.]", "", filename).strip()
    if not safe_filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    root = AUDIO_SOURCES[source].resolve()
    if source == "audiobooks":
        if not book_id:
            raise HTTPException(status_code=400,
                                detail="book_id is required for audiobook chunk audio")
        safe_book_id = re.sub(r"[^\w\s\-]", "", book_id).strip()
        candidate = (root / safe_book_id / safe_filename).resolve()
    else:
        candidate = (root / safe_filename).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403,
                            detail="Requested audio path is outside source root") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return candidate


def _shard_ready(book_id: str, role: str) -> bool:
    try:
        _load_shard_index(book_id, role)
        return True
    except HTTPException:
        return False


# ═══════════════# SECTION A — Server Lifecycle
# ═══════════════
@app.on_event("startup")
async def startup_event():
    for directory in (STATIC_DIR, TEMPLATES_DIR, INPUT_DIR, OUTPUT_DIR, AUDIOBOOKS_DIR,
                      PDF_CACHE_DIR):
        if directory in (STATIC_DIR, TEMPLATES_DIR):
            continue
        directory.mkdir(parents=True, exist_ok=True)

    pdf_health = await _downstream_health("pdf_processor", PDF_SERVICE_URL)
    if pdf_health == "ok":
        logger.info(f"PDF processor health ok at {PDF_SERVICE_URL}")
    else:
        logger.warning(f"PDF processor health degraded at {PDF_SERVICE_URL}: {pdf_health}")

    logger.info(f"Serving audio from defined sources: {list(AUDIO_SOURCES.keys())}")
    logger.info("Audio server started successfully.")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("HTTP client closed.")


@app.get("/health")
async def health():
    checks = {
        "static_dir": "ok" if STATIC_DIR.exists() else f"error: missing {STATIC_DIR}",
        "templates_dir": "ok" if TEMPLATES_DIR.exists() else f"error: missing {TEMPLATES_DIR}",
        "input_dir": "ok" if INPUT_DIR.exists() else f"error: missing {INPUT_DIR}",
        "output_dir": "ok" if OUTPUT_DIR.exists() else f"error: missing {OUTPUT_DIR}",
        "audiobooks_dir": "ok" if AUDIOBOOKS_DIR.exists() else f"error: missing {AUDIOBOOKS_DIR}",
        "pdf_cache_dir": "ok" if PDF_CACHE_DIR.exists() else f"error: missing {PDF_CACHE_DIR}",
        "pdf_processor": await _downstream_health("pdf_processor", PDF_SERVICE_URL),
        "ui_pdfjs": "ok" if UI_VENDOR_ASSETS[
            "pdfjs"].exists() else f"missing: {UI_VENDOR_ASSETS['pdfjs']}",
        "ui_pdfjs_worker": "ok" if UI_VENDOR_ASSETS[
            "pdfjs_worker"].exists() else f"missing: {UI_VENDOR_ASSETS['pdfjs_worker']}",
        "ui_wavesurfer": "ok" if UI_VENDOR_ASSETS[
            "wavesurfer"].exists() else f"missing: {UI_VENDOR_ASSETS['wavesurfer']}",
    }
    return build_health_response(
        service="tts_audio_server",
        role="audio_api",
        checks=checks,
        details={
            "pdf_service_url": PDF_SERVICE_URL,
            "acquire_service_url": ACQUIRE_SERVICE_URL,
            "convert_service_url": CONVERT_SERVICE_URL,
            "audio_sources": {name: str(path) for name, path in AUDIO_SOURCES.items()},
        },
    )


@app.get("/api/v1/ui/runtime-dependencies")
async def ui_runtime_dependencies():
    return _local_ui_dependency_status()


# ═══════════════# SECTION B — Frontend Shell / Static UI
# ═══════════════
@app.get("/", response_class=HTMLResponse)
async def get_player_interface():
    player_html_path = TEMPLATES_DIR / "player.html"
    if not player_html_path.exists():
        logger.error(f"FATAL: player.html not found at {player_html_path}")
        raise HTTPException(status_code=500, detail="Player interface file missing")
    return FileResponse(player_html_path)


# ═══════════════# SECTION C — Source & Asset Enumeration
# ═══════════════
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
            if source == "audiobooks":
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
            else:
                for f in target_directory.glob("*.wav"):
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "book_id": None,
                        "size_bytes": f.stat().st_size,
                        "type": source
                    })
        except Exception as e:
            logger.error(f"Error scanning directory {target_directory}: {e}")

    return {"files": files, "source": source}


@app.get("/api/audio/{filename}")
async def serve_audio_file(
        filename: str,
        source: Optional[str] = Query(DEFAULT_AUDIO_SOURCE_NAME),
        book_id: Optional[str] = Query(default=None),
):
    """Serve standalone/obsidian WAV files and explicitly addressed audiobook chunks."""
    source_name = source or DEFAULT_AUDIO_SOURCE_NAME
    file_path = _safe_audio_source_file(source_name, filename, book_id=book_id)
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/available_pdfs")
async def list_available_pdfs():
    """List PDFs available for processing."""
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


# ═══════════════# SECTION D — Audiobook Status APIs & Metadata
# ═══════════════
@app.get("/api/audiobooks")
async def list_audiobooks():
    """Lists all available audiobooks with their processing status"""
    books = []
    if not AUDIOBOOKS_DIR.exists():
        return {"audiobooks": books}

    for book_dir in AUDIOBOOKS_DIR.iterdir():
        if book_dir.is_dir():
            manifest_path = book_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        total_chunks = manifest.get('total_chunks', 0)
                        ready_chunks = len(manifest.get('ready_chunks', []))

                        safe_book_id = book_dir.name
                        artifact_catalog = load_audiobook_artifacts(book_dir) or {}
                        citation_artifact = find_artifact_by_role(book_dir, "citation_ready")
                        ui_artifact = find_artifact_by_role(book_dir, "ui_sentences")
                        citation_filename = safe_book_id + '_citation_ready.json'
                        highlighting_ready = bool(citation_artifact) or (
                                PDF_CACHE_DIR / citation_filename).exists()

                        job_snapshot = _load_job_for_manifest(manifest) or {}

                        books.append({
                            "book_id": book_dir.name,
                            "document_id": manifest.get('document_id') or artifact_catalog.get(
                                'document_id'),
                            "job_id": manifest.get('job_id') or artifact_catalog.get('job_id'),
                            "audiobook_id": manifest.get('audiobook_id') or artifact_catalog.get(
                                'audiobook_id') or book_dir.name,
                            "title": manifest['metadata'].get('title', book_dir.name),
                            "author": manifest['metadata'].get('author', 'Unknown'),
                            "source_file": manifest['metadata'].get('source_filename', ''),
                            "total_chunks": total_chunks,
                            "ready_chunks": ready_chunks,
                            "is_complete": ready_chunks == total_chunks and total_chunks > 0,
                            "highlighting_ready": highlighting_ready,
                            "ui_ready": bool(ui_artifact),
                            "artifact_count": len(artifact_catalog.get('artifacts', []) or []),
                            "processing_status": manifest.get('processing_status'),
                            "job_status": job_snapshot.get('status'),
                            "job_stage": job_snapshot.get('stage'),
                        })
                except Exception as e:
                    logger.error(f"Failed to read manifest for {book_dir.name}: {e}")
    return {"audiobooks": books}


@app.get("/api/v1/audiobooks")
async def list_audiobooks_v1():
    return await list_audiobooks()


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

        # Validate and normalize status fields for response projection.
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

        # Stale job detection for visible operator status.
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

        artifact_catalog = load_audiobook_artifacts(AUDIOBOOKS_DIR / safe_book_id) or {}
        response_data.setdefault('document_id', artifact_catalog.get('document_id'))
        response_data.setdefault('job_id', artifact_catalog.get('job_id'))
        response_data.setdefault('audiobook_id',
                                 artifact_catalog.get('audiobook_id') or safe_book_id)
        response_data['artifact_count'] = len(artifact_catalog.get('artifacts', []) or [])
        job_snapshot = _load_job_for_manifest(response_data)
        if job_snapshot:
            response_data['job'] = job_snapshot
            response_data['job_status'] = job_snapshot.get('status')
            response_data['job_stage'] = job_snapshot.get('stage')
            response_data['job_progress_current'] = job_snapshot.get('progress_current')
            response_data['job_progress_total'] = job_snapshot.get('progress_total')

        citation_artifact = find_artifact_by_role(AUDIOBOOKS_DIR / safe_book_id, "citation_ready")
        citation_filename = safe_book_id + '_citation_ready.json'
        citation_path = PDF_CACHE_DIR / citation_filename
        response_data['highlighting_ready'] = bool(citation_artifact) or citation_path.exists()

        ui_artifact = find_artifact_by_role(AUDIOBOOKS_DIR / safe_book_id, "ui_sentences")
        ui_sentences_filename = safe_book_id + '_ui_sentences.json'
        ui_sentences_path = PDF_CACHE_DIR / ui_sentences_filename
        response_data['ui_ready'] = bool(ui_artifact) or ui_sentences_path.exists()
        response_data['ui_shards_ready'] = _shard_ready(safe_book_id, "ui_sentences")
        response_data['semantic_shards_ready'] = _shard_ready(safe_book_id, "semantic")
        audio_timing_artifact = find_artifact_by_role(AUDIOBOOKS_DIR / safe_book_id, "audio_timing")
        audio_timing_path = AUDIOBOOKS_DIR / safe_book_id / "audio_timing.json"
        response_data['audio_timing_ready'] = bool(
            audio_timing_artifact) or audio_timing_path.exists()
        response_data['shard_urls'] = {
            "ui_sentences_index": f"/api/v1/audiobooks/{safe_book_id}/ui-sentences/pages/index",
            "semantic_index": f"/api/v1/audiobooks/{safe_book_id}/semantic/pages/index",
            "audio_timing": f"/api/v1/audiobooks/{safe_book_id}/audio-timing",
        }

        return response_data

    except Exception as e:
        logger.error(f"Error reading live manifest for {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read audiobook data")


@app.get("/api/v1/audiobooks/{book_id}/status")
async def get_audiobook_status_v1(book_id: str):
    return await get_audiobook_status(book_id)


@app.get("/api/v1/audiobooks/{book_id}/manifest")
async def get_audiobook_manifest_v1(book_id: str):
    return await get_audiobook_status(book_id)


@app.get("/api/v1/audiobooks/{book_id}/artifacts")
async def get_audiobook_artifacts(book_id: str):
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    catalog = load_audiobook_artifacts(AUDIOBOOKS_DIR / safe_book_id)
    if not catalog:
        raise HTTPException(status_code=404,
                            detail=f"Artifact catalog not found for audiobook '{book_id}'")
    return catalog


@app.get("/api/v1/traces/{trace_id}/artifacts")
async def get_trace_artifacts(trace_id: str):
    catalog = load_trace_catalog(trace_id, cache_dir=PDF_CACHE_DIR)
    if not catalog:
        raise HTTPException(status_code=404,
                            detail=f"Artifact catalog not found for trace '{trace_id}'")
    return catalog


@app.get("/api/v1/traces/{trace_id}/jobs")
async def get_trace_jobs_v1(trace_id: str):
    return load_trace_jobs(trace_id, cache_dir=PDF_CACHE_DIR)


@app.get("/api/v1/audiobooks/{book_id}/job")
async def get_audiobook_job_v1(book_id: str):
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()
    manifest_path = AUDIOBOOKS_DIR / safe_book_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Audiobook manifest not found: {book_id}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    job = _load_job_for_manifest(manifest)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found for audiobook: {book_id}")
    return job


@app.get("/api/v1/documents/{document_id}")
async def get_document_asset(document_id: str):
    asset = load_document_asset(document_id, cache_dir=PDF_CACHE_DIR)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Document asset not found: {document_id}")
    return asset


@app.get("/api/v1/documents/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    asset = load_document_asset(document_id, cache_dir=PDF_CACHE_DIR)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Document asset not found: {document_id}")
    canonical_pdf = asset.get("canonical_pdf") or {}
    return _artifact_file_response(canonical_pdf, fallback_media_type="application/pdf")


@app.get("/api/v1/artifacts/{artifact_id}")
async def get_artifact_file(artifact_id: str):
    artifact = find_artifact_by_id(artifact_id, cache_dir=PDF_CACHE_DIR, output_dir=AUDIOBOOKS_DIR)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    return _artifact_file_response(artifact)


@app.get("/api/v1/jobs")
async def list_jobs_v1(
        status: Optional[str] = None,
        book_id: Optional[str] = None,
        limit: int = 100,
):
    return {"jobs": list_jobs(cache_dir=PDF_CACHE_DIR, status=status, book_id=book_id, limit=limit)}


@app.get("/api/v1/jobs/{job_id}")
async def get_job_v1(job_id: str):
    job = load_job(job_id, cache_dir=PDF_CACHE_DIR)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.get("/api/v1/jobs/{job_id}/events")
async def get_job_events_v1(job_id: str, limit: int = 500):
    if not load_job(job_id, cache_dir=PDF_CACHE_DIR):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"job_id": job_id,
            "events": list_job_events(job_id, cache_dir=PDF_CACHE_DIR, limit=limit)}


@app.get("/api/v1/audiobooks/{book_id}/audio-timing")
async def get_audiobook_audio_timing_v1(book_id: str):
    timing_artifact = _artifact_for_book_role(book_id, "audio_timing")
    if timing_artifact:
        return _artifact_file_response(timing_artifact, fallback_media_type="application/json")
    timing_path = AUDIOBOOKS_DIR / _safe_book_dirname(book_id) / "audio_timing.json"
    if not timing_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio timing artifact not found: {book_id}")
    return FileResponse(timing_path, media_type="application/json")


@app.get("/api/v1/audiobooks/{book_id}/stage3-checkpoint")
async def get_audiobook_stage3_checkpoint_v1(book_id: str):
    checkpoint_artifact = _artifact_for_book_role(book_id, "stage3_checkpoint")
    if checkpoint_artifact:
        return _artifact_file_response(checkpoint_artifact, fallback_media_type="application/json")
    checkpoint_path = AUDIOBOOKS_DIR / _safe_book_dirname(book_id) / "stage3_checkpoint.json"
    checkpoint_path = _safe_workspace_path(str(checkpoint_path))
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"Stage 3 checkpoint not found for audiobook '{book_id}'")
    return FileResponse(path=checkpoint_path, media_type="application/json")


@app.get("/api/v1/audiobooks/{book_id}/stage3/checkpoint")
async def get_audiobook_stage3_checkpoint_canonical_v1(book_id: str):
    return await get_audiobook_stage3_checkpoint_v1(book_id)


@app.get("/api/v1/audiobooks/{book_id}/stage3/chunks/{chunk_id}")
async def get_audiobook_stage3_chunk_checkpoint_v1(book_id: str, chunk_id: int):
    if chunk_id < 0:
        raise HTTPException(status_code=422, detail="chunk_id must be non-negative")
    checkpoint_path = AUDIOBOOKS_DIR / _safe_book_dirname(
        book_id) / "stage3_chunks" / f"chunk_{chunk_id:06d}.json"
    checkpoint_path = _safe_workspace_path(str(checkpoint_path))
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise HTTPException(status_code=404,
                            detail=f"Stage 3 chunk checkpoint not found: {chunk_id}")
    return FileResponse(path=checkpoint_path, media_type="application/json")


# ═══════════════# SECTION E — UI Sentence & Citation APIs
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

    ui_artifact = _artifact_for_book_role(safe_book_id, "ui_sentences")
    if ui_artifact:
        logger.info(f"Serving UI sentences artifact for: {book_id}")
        return _artifact_file_response(ui_artifact, fallback_media_type="application/json")

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


@app.get("/api/v1/audiobooks/{book_id}/ui-sentences")
async def get_audiobook_ui_sentences_v1(book_id: str):
    return await get_audiobook_ui_sentences(book_id)


@app.get("/api/v1/audiobooks/{book_id}/ui-sentences/index")
async def get_audiobook_ui_sentences_index_v1(book_id: str):
    return _shard_index_response(book_id, "ui_sentences")


@app.get("/api/v1/audiobooks/{book_id}/ui-sentences/pages/index")
async def get_audiobook_ui_sentences_pages_index_v1(book_id: str):
    return _shard_index_response(book_id, "ui_sentences")


@app.get("/api/v1/audiobooks/{book_id}/ui-sentences/pages/{page_number}")
async def get_audiobook_ui_sentences_page_v1(book_id: str, page_number: int):
    return _shard_page_response(book_id, "ui_sentences", page_number)


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

    semantic_artifact = _artifact_for_book_role(safe_book_id, "semantic")
    if semantic_artifact:
        logger.info(f"Serving semantic artifact for: {book_id}")
        return _artifact_file_response(semantic_artifact, fallback_media_type="application/json")

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


@app.get("/api/v1/audiobooks/{book_id}/semantic")
async def get_audiobook_semantic_v1(book_id: str):
    return await get_audiobook_semantic(book_id)


@app.get("/api/v1/audiobooks/{book_id}/semantic/index")
async def get_audiobook_semantic_index_v1(book_id: str):
    return _shard_index_response(book_id, "semantic")


@app.get("/api/v1/audiobooks/{book_id}/semantic/pages/index")
async def get_audiobook_semantic_pages_index_v1(book_id: str):
    return _shard_index_response(book_id, "semantic")


@app.get("/api/v1/audiobooks/{book_id}/semantic/pages/{page_number}")
async def get_audiobook_semantic_page_v1(book_id: str, page_number: int):
    return _shard_page_response(book_id, "semantic", page_number)


# ═══════════════# SECTION F — Audio Chunk Serving
# ═══════════════
async def serve_audiobook_chunk(book_id: str, chunk_filename: str):
    """Serve a specific audio chunk from an audiobook."""
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


@app.get("/api/v1/audiobooks/{book_id}/chunks/{chunk_filename}/audio")
async def serve_audiobook_chunk_v1(book_id: str, chunk_filename: str):
    return await serve_audiobook_chunk(book_id, chunk_filename)


# ═══════════════# SECTION G — PDF Proxy APIs
# ═══════════════
@app.get("/api/pdf/{pdf_filename}")
async def proxy_serve_pdf(pdf_filename: str):
    """Proxy request for source PDF document to the pdf-processor service."""
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

    except httpx.TimeoutException:
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
async def start_pdf_processing(pdf_filename: str, raw_request: Request):
    """
    Proxy PDF processing request to pdf-processor service.
    Delegates full pipeline execution to the processor service.
    """
    safe_filename = re.sub(r'[^\w\s\-\.]', '', pdf_filename).strip()

    if not safe_filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file")

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/process/{safe_filename}"
        logger.info(f"Proxying process request: {api_url}")

        forwarded_headers = {}
        for header_name in ("X-Trace-ID", "X-Document-ID", "X-Job-ID", "X-Audiobook-ID"):
            value = raw_request.headers.get(header_name)
            if value:
                forwarded_headers[header_name] = value

        response = await client.post(api_url, headers=forwarded_headers or None)
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


# ═══════════════# SECTION H — Processing Control APIs
# ═══════════════
async def retry_processing(book_id: str, force_rebuild: bool = False):
    """
    Proxy retry request to pdf-processor service.
    """
    safe_book_id = re.sub(r'[^\w\s\-]', '', book_id).strip()

    try:
        api_url = f"{PDF_SERVICE_URL}/api/v1/retry/{safe_book_id}"

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


@app.post("/api/v1/audiobooks/{book_id}/retry")
async def retry_processing_v1(book_id: str, force_rebuild: bool = False):
    return await retry_processing(book_id, force_rebuild=force_rebuild)


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


@app.post("/api/v1/audiobooks/{book_id}/rebuild-selective")
async def rebuild_selective_v1(
        book_id: str,
        chunk_ids: Optional[str] = None,
        pages: Optional[str] = None,
):
    return await rebuild_selective(book_id, chunk_ids=chunk_ids, pages=pages)


# ═══════════════
# SECTION I — Ingest (Gateway Orchestration Entry Point)


async def _run_ingest(
        *,
        url: Optional[str],
        file: Optional[UploadFile],
        trace_id: Optional[str],
):
    try:
        result = await route_ingest(url=url, file=file, supplied_trace_id=trace_id)
        return result.model_dump()
    except ValueError as e:
        logger.error(f"Ingest routing error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"Ingest downstream HTTP error: {e.response.status_code}")
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error(f"Ingest unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Ingest failed")


@app.post("/api/v1/ingest/url")
async def ingest_url(body: IngestUrlRequest = Body(...)):
    return await _run_ingest(url=body.url, file=None, trace_id=body.trace_id)


@app.post("/api/v1/ingest/file")
async def ingest_file(
        file: UploadFile = File(...),
        trace_id: Optional[str] = Form(default=None),
):
    return await _run_ingest(url=None, file=file, trace_id=trace_id)