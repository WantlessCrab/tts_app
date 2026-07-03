# ~/TTS/my_app/gateway_router.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY ROUTER — Routing decisions and orchestration sequence
# Generates/preserves trace_id. Classifies input. Calls correct client. Writes
# canonical PDF. Triggers processor. Returns braid-ready GatewayResult.
# No HTTP handling here — that lives in gateway_clients.py.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from .artifact_store import (
    build_artifact_ref_from_path,
    document_id_for_bytes,
    register_artifact,
    write_bytes_atomic,
    write_document_asset,
)
from .gateway_contracts import GatewayResult
from .gateway_clients import call_acquire, call_convert, call_processor
from .job_store import append_job_event, update_job_state, write_job
from .toolset_contracts import (
    AudiobookJob,
    DocumentAsset,
    audiobook_id_from_document_id,
    model_to_dict,
    new_job_id,
    new_trace_id,
    sha256_bytes,
)

logger = logging.getLogger("GatewayRouter")

WORKSPACE_DIR = Path("/workspace")
INPUT_DIR = WORKSPACE_DIR / "pdf_input"
CACHE_DIR = WORKSPACE_DIR / "pdf_cache"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "audiobooks"


def _is_pdf_bytes(data: bytes) -> bool:
    return bool(data) and data.startswith(b"%PDF-")


def _source_kind_for_upload(filename: str, content_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if content_type == "application/pdf" or ext == ".pdf":
        return "upload_pdf"
    if ext in {".epub", ".mobi", ".azw", ".azw3"}:
        return "upload_ebook"
    if ext in {".docx", ".doc", ".odt", ".pptx", ".ppt", ".odp", ".xlsx", ".xls", ".ods", ".rtf"}:
        return "upload_office"
    return "upload_other"


async def route_ingest(
        url: Optional[str] = None,
        file: Optional[UploadFile] = None,
        supplied_trace_id: Optional[str] = None,
) -> GatewayResult:
    """
    Single orchestration entry point for all ingest paths.

    Path A: URL provided           → call_acquire → write PDF → call_processor
    Path B: Office/Ebook file      → call_convert → write PDF → call_processor
    Path C: PDF file uploaded      → write PDF directly → call_processor

    Raises ValueError on invalid input or non-PDF response from any service.
    """
    trace_id = supplied_trace_id or new_trace_id()
    source_filename: Optional[str] = None
    source_uri: Optional[str] = None
    source_kind = "upload_other"
    source_provenance: dict[str, object] = {}

    # ── Path A: URL ───────────────────────────────────────────────────────────
    if url:
        source_uri = url
        source_filename = url
        source_kind = "url"
        logger.info(f"[{trace_id}] Route: URL → acquire")
        acquired = await call_acquire(url, trace_id)
        pdf_bytes = acquired.content
        source_provenance = {
            "source_kind": source_kind,
            "source_uri": source_uri,
            "acquire_method": acquired.method,
            "acquire_source": acquired.source,
            "downstream_headers": acquired.headers or {},
        }

    # ── Path B / C: File upload ───────────────────────────────────────────────
    elif file:
        source_filename = file.filename or "upload"
        file_bytes = await file.read()
        content_type = (file.content_type or "").lower()
        ext = Path(source_filename).suffix.lower()
        source_kind = _source_kind_for_upload(source_filename, content_type)

        if content_type == "application/pdf" or ext == ".pdf":
            if not _is_pdf_bytes(file_bytes):
                raise ValueError(f"[{trace_id}] Uploaded file is not a valid PDF")
            logger.info(f"[{trace_id}] Route: direct PDF upload → write")
            pdf_bytes = file_bytes
            source_provenance = {
                "source_kind": source_kind,
                "source_filename": source_filename,
                "content_type": content_type,
                "ingest_method": "direct_pdf_upload",
            }
        else:
            logger.info(f"[{trace_id}] Route: {content_type or ext or 'unknown'} → convert")
            converted = await call_convert(file_bytes, source_filename, trace_id)
            pdf_bytes = converted.content
            source_provenance = {
                "source_kind": source_kind,
                "source_filename": source_filename,
                "content_type": content_type,
                "conversion_method": converted.method,
                "conversion_source": converted.source,
                "downstream_headers": converted.headers or {},
            }

    else:
        raise ValueError("Must provide either url or file")

    # ── Canonical identity ────────────────────────────────────────────────────
    if not _is_pdf_bytes(pdf_bytes):
        raise ValueError(f"[{trace_id}] Ingest pipeline did not produce valid PDF bytes")

    if not source_provenance:
        source_provenance = {
            "source_kind": source_kind,
            "source_filename": source_filename,
            "source_uri": source_uri,
        }

    content_sha256 = sha256_bytes(pdf_bytes)
    document_id = document_id_for_bytes(pdf_bytes)
    audiobook_id = audiobook_id_from_document_id(document_id)
    job_id = new_job_id()
    pdf_filename = f"{document_id}.pdf"
    pdf_path = INPUT_DIR / pdf_filename
    audiobook_dir = OUTPUT_DIR / audiobook_id

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    audiobook_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{trace_id}] Writing canonical PDF {len(pdf_bytes) // 1024} KB → {pdf_path} "
        f"[document_id={document_id} audiobook_id={audiobook_id} job_id={job_id}]"
    )
    write_bytes_atomic(pdf_path, pdf_bytes)

    canonical_pdf_ref = build_artifact_ref_from_path(
        path=pdf_path,
        role="canonical_pdf",
        trace_id=trace_id,
        job_id=job_id,
        document_id=document_id,
        audiobook_id=audiobook_id,
        mime_type="application/pdf",
        schema_version="tts.canonical_pdf.v1",
        metadata={
            "source_kind": source_kind,
            "source_filename": source_filename,
            "source_uri": source_uri,
            "source_provenance": source_provenance,
        },
    )
    register_artifact(canonical_pdf_ref, cache_dir=CACHE_DIR, audiobook_dir=audiobook_dir)

    document_asset = DocumentAsset(
        document_id=document_id,
        trace_id=trace_id,
        source_kind=source_kind,  # type: ignore[arg-type]
        source_uri=source_uri,
        source_filename=source_filename,
        content_sha256=content_sha256,
        canonical_pdf=canonical_pdf_ref,
        metadata={
            "gateway_pdf_filename": pdf_filename,
            "source_provenance": source_provenance,
        },
    )
    write_document_asset(document_asset, cache_dir=CACHE_DIR)

    job = AudiobookJob(
        job_id=job_id,
        trace_id=trace_id,
        document_id=document_id,
        audiobook_id=audiobook_id,
        status="processing_started",
        stage="processor_queued",
        progress_current=0,
        progress_total=0,
        artifacts=[canonical_pdf_ref],
        metadata={
            "source_filename": source_filename,
            "pdf_filename": pdf_filename,
            "source_kind": source_kind,
            "source_provenance": source_provenance,
        },
    )
    write_job(job, cache_dir=CACHE_DIR)
    append_job_event(
        cache_dir=CACHE_DIR,
        job_id=job_id,
        trace_id=trace_id,
        event_type="gateway_job_created",
        status=job.status,
        stage=job.stage,
        document_id=document_id,
        audiobook_id=audiobook_id,
        book_id=audiobook_id,
        message="Gateway wrote canonical PDF and queued processor handoff.",
        data={
            "source_kind": source_kind,
            "source_filename": source_filename,
            "source_provenance": source_provenance,
        },
    )

    # ── Trigger processor ─────────────────────────────────────────────────────
    result = await call_processor(
        pdf_filename,
        trace_id,
        document_id=document_id,
        job_id=job_id,
        audiobook_id=audiobook_id,
    )

    accepted_book_id = result.get("book_id", audiobook_id)
    job = AudiobookJob(**update_job_state(
        cache_dir=CACHE_DIR,
        job_id=result.get("job_id", job_id),
        trace_id=trace_id,
        document_id=result.get("document_id", document_id),
        audiobook_id=result.get("audiobook_id", audiobook_id),
        status=result.get("status", "processing_started"),
        stage="processor_accepted",
        progress_current=0,
        progress_total=0,
        artifacts=[model_to_dict(canonical_pdf_ref)],
        metadata_patch={
            "source_filename": source_filename,
            "pdf_filename": pdf_filename,
            "source_kind": source_kind,
            "source_provenance": source_provenance,
        },
        book_id=accepted_book_id,
        event_type="processor_accepted",
        message="Processor accepted the job and background execution is active.",
        event_data=result,
    ))

    return GatewayResult(
        book_id=accepted_book_id,
        trace_id=trace_id,
        status=result.get("status", "processing_started"),
        source_filename=source_filename,
        document_id=result.get("document_id", document_id),
        job_id=result.get("job_id", job_id),
        audiobook_id=result.get("audiobook_id", audiobook_id),
        canonical_pdf_artifact=canonical_pdf_ref,
        document_asset=document_asset,
        job=job,
        source_provenance=source_provenance,
    )