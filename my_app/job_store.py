# ~/TTS/my_app/job_store.py

"""Filesystem durable job state and event log for tts_app.

The ledger is file-backed by design in this phase. It gives the TTS service
restart-visible job state without taking PostgreSQL/schema authority from
data_stack before the TTS contracts are stable.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import uuid
from typing import Any, Iterable, Optional

from .artifact_store import atomic_write_json, merge_artifact_ref_dicts, read_json
from .toolset_contracts import AudiobookJob, JobEvent, model_to_dict, utc_now_iso

JOB_STATE_DIRNAME = "job_state"
JOB_EVENTS_DIRNAME = "job_events"
_ALLOWED_JOB_STATUSES = {
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
}
_TERMINAL_STATUSES = {"stage_3_complete", "completed", "failed", "cancelled", "stage_3_failed",
                      "degraded"}


def _job_state_dir(cache_dir: Path) -> Path:
    return cache_dir / JOB_STATE_DIRNAME


def _job_events_dir(cache_dir: Path) -> Path:
    return cache_dir / JOB_EVENTS_DIRNAME


def job_state_path(job_id: str, *, cache_dir: Path) -> Path:
    return _job_state_dir(cache_dir) / f"{job_id}.json"


def job_events_path(job_id: str, *, cache_dir: Path) -> Path:
    return _job_events_dir(cache_dir) / f"{job_id}.jsonl"


def _coerce_status(status: Optional[str], *, fallback: str = "running") -> str:
    candidate = (status or fallback or "running").strip()
    if candidate in _ALLOWED_JOB_STATUSES:
        return candidate
    if "fail" in candidate:
        return "failed"
    if "complete" in candidate:
        return "completed"
    if "cancel" in candidate:
        return "cancelled"
    return fallback if fallback in _ALLOWED_JOB_STATUSES else "running"


def load_job(job_id: str, *, cache_dir: Path) -> Optional[dict[str, Any]]:
    return read_json(job_state_path(job_id, cache_dir=cache_dir))


def write_job(
        job: AudiobookJob | dict[str, Any],
        *,
        cache_dir: Path,
        audiobook_dir: Optional[Path] = None,
) -> dict[str, Any]:
    payload = model_to_dict(job) if isinstance(job, AudiobookJob) else dict(job)
    payload["status"] = _coerce_status(payload.get("status"), fallback="queued")
    payload.setdefault("created_at", utc_now_iso())
    payload["updated_at"] = utc_now_iso()
    validated = AudiobookJob(**payload)
    output = model_to_dict(validated)
    atomic_write_json(job_state_path(validated.job_id, cache_dir=cache_dir), output)
    if audiobook_dir is not None:
        atomic_write_json(audiobook_dir / "job.json", output)
    return output


def append_job_event(
        *,
        cache_dir: Path,
        job_id: str,
        trace_id: str,
        event_type: str,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        document_id: Optional[str] = None,
        audiobook_id: Optional[str] = None,
        book_id: Optional[str] = None,
        message: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        progress_current: Optional[int] = None,
        progress_total: Optional[int] = None,
        error: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    event_data: dict[str, Any] = {}
    if metadata:
        event_data.update(metadata)
    if data:
        event_data.update(data)
    if progress_current is not None:
        event_data["progress_current"] = progress_current
    if progress_total is not None:
        event_data["progress_total"] = progress_total
    if error is not None:
        event_data["error"] = error

    event = JobEvent(
        event_id=f"evt_{uuid.uuid4().hex}",
        job_id=job_id,
        trace_id=trace_id,
        event_type=event_type,
        status=_coerce_status(status, fallback="running") if status else None,
        stage=stage,
        document_id=document_id,
        audiobook_id=audiobook_id,
        book_id=book_id,
        message=message,
        data=event_data,
    )
    payload = model_to_dict(event)
    path = job_events_path(job_id, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _state_changed(existing: Optional[dict[str, Any]], payload: dict[str, Any]) -> bool:
    if not existing:
        return True
    keys = ("status", "stage", "progress_current", "progress_total", "error")
    return any(existing.get(key) != payload.get(key) for key in keys)


def update_job_state(
        *,
        cache_dir: Path,
        job_id: str,
        trace_id: str,
        document_id: Optional[str],
        audiobook_id: Optional[str],
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress_current: Optional[int] = None,
        progress_total: Optional[int] = None,
        error: Optional[dict[str, Any]] = None,
        artifacts: Optional[Iterable[dict[str, Any]]] = None,
        metadata_patch: Optional[dict[str, Any]] = None,
        book_id: Optional[str] = None,
        event_type: Optional[str] = None,
        message: Optional[str] = None,
        event_data: Optional[dict[str, Any]] = None,
        suppress_event: bool = False,
) -> dict[str, Any]:
    existing = load_job(job_id, cache_dir=cache_dir) or {}
    payload = dict(existing)
    now = utc_now_iso()

    payload.setdefault("job_id", job_id)
    payload.setdefault("trace_id", trace_id)
    payload.setdefault("document_id", document_id or "unknown_document")
    payload.setdefault("audiobook_id", audiobook_id or book_id or "unknown_audiobook")
    payload.setdefault("created_at", now)

    payload["trace_id"] = trace_id or payload.get("trace_id")
    if document_id:
        payload["document_id"] = document_id
    if audiobook_id:
        payload["audiobook_id"] = audiobook_id
    payload["status"] = _coerce_status(status or payload.get("status"), fallback="running")
    payload["stage"] = stage or payload.get("stage") or payload["status"]
    payload["progress_current"] = max(0,
                                      int(progress_current if progress_current is not None else payload.get(
                                          "progress_current") or 0))
    payload["progress_total"] = max(0,
                                    int(progress_total if progress_total is not None else payload.get(
                                        "progress_total") or 0))
    if error is not None:
        payload["error"] = error
    elif payload.get("status") not in {"failed", "stage_3_failed"}:
        payload.setdefault("error", None)

    payload["artifacts"] = merge_artifact_ref_dicts(payload.get("artifacts", []), artifacts or [])
    metadata = dict(payload.get("metadata") or {})
    if metadata_patch:
        metadata.update(metadata_patch)
    if book_id:
        metadata["book_id"] = book_id
    payload["metadata"] = metadata
    payload["updated_at"] = now

    should_event = (not suppress_event) and (
                bool(event_type) or _state_changed(existing or None, payload))
    output = write_job(payload, cache_dir=cache_dir)

    if should_event:
        append_job_event(
            cache_dir=cache_dir,
            job_id=job_id,
            trace_id=output["trace_id"],
            event_type=event_type or "job_state_changed",
            status=output.get("status"),
            stage=output.get("stage"),
            document_id=output.get("document_id"),
            audiobook_id=output.get("audiobook_id"),
            book_id=output.get("metadata", {}).get("book_id"),
            message=message,
            data=event_data or {},
        )
    return output


def manifest_job_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    ready_chunks = manifest.get("ready_chunks") or []
    total_chunks = int(manifest.get("total_chunks") or 0)
    ready_count = len(ready_chunks) if isinstance(ready_chunks, list) else 0
    processing_status = _coerce_status(manifest.get("processing_status"), fallback="running")
    percentage = round((ready_count / total_chunks) * 100, 1) if total_chunks > 0 else 0.0
    return {
        "job_status": processing_status,
        "job_stage": str(manifest.get("processing_status") or processing_status),
        "progress_current": ready_count,
        "progress_total": total_chunks,
        "progress_percentage": percentage,
    }


def sync_job_from_manifest(
        manifest: dict[str, Any],
        *,
        cache_dir: Path,
        event_type: Optional[str] = None,
        message: Optional[str] = None,
        audiobook_dir: Optional[Path] = None,
        event_data: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    job_id = manifest.get("job_id")
    if not job_id:
        return None
    ready_chunks = manifest.get("ready_chunks") or []
    total_chunks = int(manifest.get("total_chunks") or 0)
    status = _coerce_status(manifest.get("processing_status"), fallback="running")
    error_message = manifest.get("error_message")
    error = {"message": error_message} if error_message else None
    metadata = dict(manifest.get("metadata") or {})
    metadata["book_id"] = manifest.get("book_id")
    metadata["is_terminal"] = status in _TERMINAL_STATUSES
    metadata["ready_chunk_count"] = len(ready_chunks)
    output = update_job_state(
        cache_dir=cache_dir,
        job_id=str(job_id),
        trace_id=str(manifest.get("trace_id") or ""),
        document_id=manifest.get("document_id"),
        audiobook_id=manifest.get("audiobook_id") or manifest.get("book_id"),
        status=status,
        stage=str(manifest.get("processing_status") or status),
        progress_current=len(ready_chunks),
        progress_total=total_chunks,
        error=error,
        artifacts=manifest.get("artifact_refs") or [],
        metadata_patch=metadata,
        book_id=manifest.get("book_id"),
        event_type=event_type,
        message=message,
        event_data=event_data,
        suppress_event=(event_type is None),
    )
    if output and audiobook_dir is not None:
        atomic_write_json(audiobook_dir / "job.json", output)
    return output


def list_job_events(job_id: str, *, cache_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    path = job_events_path(job_id, cache_dir=cache_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict):
                events.append(event)
    if limit and limit > 0:
        return events[-limit:]
    return events


def list_jobs(
        *,
        cache_dir: Path,
        status: Optional[str] = None,
        book_id: Optional[str] = None,
        limit: int = 100,
) -> list[dict[str, Any]]:
    root = _job_state_dir(cache_dir)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        data = read_json(path)
        if not data:
            continue
        if status and data.get("status") != status:
            continue
        if book_id and data.get("metadata", {}).get("book_id") != book_id:
            continue
        rows.append(data)
    rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""),
              reverse=True)
    return rows[:max(1, int(limit or 100))]


def load_trace_jobs(trace_id: str, *, cache_dir: Path, limit: int = 100) -> dict[str, Any]:
    jobs = [job for job in list_jobs(cache_dir=cache_dir, limit=limit) if
            job.get("trace_id") == trace_id]
    return {"trace_id": trace_id, "jobs": jobs}