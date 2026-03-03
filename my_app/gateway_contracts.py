# ~/TTS/my_app/gateway_contracts.py

# ════════════════════════════════════════════════════════════════════════════════
# GATEWAY CONTRACTS — Pydantic boundary models
# All data crossing a gateway service boundary is validated here.
# No business logic. No imports from other gateway modules.
# ════════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, AnyHttpUrl
from typing import Optional


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
    """
    book_id: str
    trace_id: str
    status: str
    source_filename: Optional[str] = None