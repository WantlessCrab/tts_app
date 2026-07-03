# ~/TTS/my_app/gateway_contracts.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY CONTRACTS — Pydantic boundary models
# All data crossing a gateway service boundary is validated here.
# No business logic. No imports from other gateway modules.
# ════════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Optional

from pydantic import AnyHttpUrl, BaseModel

from .toolset_contracts import ArtifactRef, DocumentAsset, AudiobookJob


# ─── INBOUND: client → gateway ───────────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Validated inbound ingestion request.
    Either:
      • url (web acquisition path)
      • file upload (handled separately by FastAPI UploadFile)
    """
    url: Optional[AnyHttpUrl] = None
    trace_id: Optional[str] = None


# ─── OUTBOUND: gateway → acquire service ─────────────────────────────────────

class AcquireRequest(BaseModel):
    """Contract for gateway → acquire service call."""
    url: AnyHttpUrl
    trace_id: str


# ─── OUTBOUND: gateway → convert service ─────────────────────────────────────

class ConvertRequest(BaseModel):
    """Contract for gateway → convert service call."""
    filename: str
    trace_id: str


# ─── OUTBOUND: gateway → client ──────────────────────────────────────────────

class GatewayResult(BaseModel):
    """
    Response returned to client after ingestion routing begins.

    book_id is the audiobook directory identifier used by player and audio routes.
    document_id, job_id, audiobook_id, and artifact refs are the durable
    lineage identities used by the canonical API surface.
    """
    book_id: str
    trace_id: str
    status: str
    source_filename: Optional[str] = None
    document_id: Optional[str] = None
    job_id: Optional[str] = None
    audiobook_id: Optional[str] = None
    canonical_pdf_artifact: Optional[ArtifactRef] = None
    document_asset: Optional[DocumentAsset] = None
    job: Optional[AudiobookJob] = None
    source_provenance: Optional[dict[str, Any]] = None