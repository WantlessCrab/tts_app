# ~/TTS/my_app/toolset_contracts.py

"""Shared tts_app identity, artifact, and job contracts.

These models describe cross-service facts only. They do not perform routing,
processing, storage I/O, database work, browser work, PDF extraction, or TTS
synthesis.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

OWNER_SYSTEM = "tts_app"
PRIVACY_LEVELS = Literal["none", "standard", "strict"]
JOB_STATUSES = Literal[
    "queued",
    "running",
    "processing_started",
    "stage_1_extracting",
    "stage_1_complete",
    "stage_2_semantic",
    "stage_2_complete",
    "stage_25_ui",
    "stage_25_complete",
    "stage_3_started",
    "stage_3_running",
    "stage_3_partial",
    "stage_3_complete",
    "stage_3_failed",
    "completed",
    "failed",
    "cancelled",
    "degraded",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    """Return a UUID string compatible with existing trace_id consumers."""
    return str(uuid.uuid4())


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, digest: str, length: int = 24) -> str:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix).strip("_") or "id"
    return f"{safe_prefix}_{digest[:length]}"


def document_id_from_sha256(digest: str) -> str:
    return stable_id("doc", digest)


def audiobook_id_from_document_id(document_id: str, profile: str = "default") -> str:
    source = f"{document_id}:{profile}".encode("utf-8")
    return stable_id("ab", hashlib.sha256(source).hexdigest())


def artifact_id_from_role_and_sha256(role: str, digest: str) -> str:
    safe_role = re.sub(r"[^a-zA-Z0-9_]+", "_", role).strip("_") or "artifact"
    return stable_id(f"art_{safe_role}", digest)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


class TraceContext(BaseModel):
    """Lineage envelope for one local tts_app workflow."""

    trace_id: str = Field(default_factory=new_trace_id)
    parent_trace_id: Optional[str] = None
    source_system: str = OWNER_SYSTEM
    source_record_id: Optional[str] = None
    privacy_level: PRIVACY_LEVELS = "none"
    created_at: str = Field(default_factory=utc_now_iso)


class ArtifactRef(BaseModel):
    """Durable identity and integrity record for one file artifact."""

    artifact_id: str
    owner_system: str = OWNER_SYSTEM
    trace_id: str
    job_id: Optional[str] = None
    document_id: Optional[str] = None
    audiobook_id: Optional[str] = None
    role: str
    path: str
    mime_type: str
    sha256: str
    size_bytes: int
    schema_version: str
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentAsset(BaseModel):
    """Canonical source document identity and its primary PDF artifact."""

    document_id: str
    trace_id: str
    source_kind: Literal["url", "upload_pdf", "upload_office", "upload_ebook", "upload_other"]
    source_uri: Optional[str] = None
    source_filename: Optional[str] = None
    content_sha256: str
    canonical_pdf: ArtifactRef
    extraction_artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudiobookJob(BaseModel):
    """Durable execution identity for one audiobook processing attempt."""

    job_id: str
    trace_id: str
    document_id: str
    audiobook_id: str
    status: JOB_STATUSES = "queued"
    stage: str = "queued"
    progress_current: int = 0
    progress_total: int = 0
    error: Optional[dict[str, Any]] = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobEvent(BaseModel):
    """Append-only event describing a visible job transition or progress checkpoint."""

    event_id: str
    job_id: str
    trace_id: str
    event_type: str
    status: Optional[JOB_STATUSES] = None
    stage: Optional[str] = None
    document_id: Optional[str] = None
    audiobook_id: Optional[str] = None
    book_id: Optional[str] = None
    message: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)