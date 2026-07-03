"""Filesystem artifact catalog for tts_app.

The catalog is intentionally file-backed in this phase. It gives the app stable
trace/document/job/audiobook/artifact identity without taking schema authority
from data_stack or requiring a PostgreSQL migration before the TTS service is
ready.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import tempfile
from typing import Any, Iterable, Optional

from .toolset_contracts import (
    ArtifactRef,
    DocumentAsset,
    artifact_id_from_role_and_sha256,
    document_id_from_sha256,
    model_to_dict,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)

TRACE_CATALOG_DIRNAME = "artifact_catalog"
DOCUMENT_CATALOG_DIRNAME = "documents"
AUDIOBOOK_ARTIFACTS_FILENAME = "artifacts.json"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Optional[Path] = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.move(str(tmp), str(path))
    except Exception:
        if tmp and tmp.exists():
            tmp.unlink()
        raise


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Optional[Path] = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.move(str(tmp), str(path))
    except Exception:
        if tmp and tmp.exists():
            tmp.unlink()
        raise


def document_id_for_bytes(data: bytes) -> str:
    return document_id_from_sha256(sha256_bytes(data))


def document_id_for_file(path: Path) -> str:
    return document_id_from_sha256(sha256_file(path))


def build_artifact_ref_from_path(
        *,
        path: Path,
        role: str,
        trace_id: str,
        mime_type: str,
        schema_version: str,
        job_id: Optional[str] = None,
        document_id: Optional[str] = None,
        audiobook_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
) -> ArtifactRef:
    digest = sha256_file(path)
    stat = path.stat()
    return ArtifactRef(
        artifact_id=artifact_id_from_role_and_sha256(role, digest),
        trace_id=trace_id,
        job_id=job_id,
        document_id=document_id,
        audiobook_id=audiobook_id,
        role=role,
        path=str(path),
        mime_type=mime_type,
        sha256=digest,
        size_bytes=stat.st_size,
        schema_version=schema_version,
        metadata=metadata or {},
    )


def _artifact_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("artifact_id") or ""),
        str(item.get("role") or ""),
        str(item.get("path") or ""),
    )


def merge_artifact_ref_dicts(
        existing: Optional[Iterable[dict[str, Any]]],
        incoming: Iterable[dict[str, Any] | ArtifactRef],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in existing or []:
        if isinstance(item, dict):
            merged[_artifact_key(item)] = item
    for item in incoming:
        payload = model_to_dict(item) if isinstance(item, ArtifactRef) else dict(item)
        merged[_artifact_key(payload)] = payload
    return sorted(
        merged.values(),
        key=lambda row: (str(row.get("role") or ""), str(row.get("created_at") or ""),
                         str(row.get("path") or "")),
    )


def _catalog_path(cache_dir: Path, trace_id: str) -> Path:
    return cache_dir / TRACE_CATALOG_DIRNAME / f"{trace_id}.json"


def _document_path(cache_dir: Path, document_id: str) -> Path:
    return cache_dir / DOCUMENT_CATALOG_DIRNAME / f"{document_id}.json"


def _audiobook_artifacts_path(audiobook_dir: Path) -> Path:
    return audiobook_dir / AUDIOBOOK_ARTIFACTS_FILENAME


def register_artifact(
        ref: ArtifactRef,
        *,
        cache_dir: Path,
        audiobook_dir: Optional[Path] = None,
) -> ArtifactRef:
    payload = model_to_dict(ref)

    trace_path = _catalog_path(cache_dir, ref.trace_id)
    trace_catalog = read_json(trace_path) or {
        "schema_version": "tts.artifact_catalog.v1",
        "trace_id": ref.trace_id,
        "created_at": utc_now_iso(),
        "artifacts": [],
    }
    trace_catalog["updated_at"] = utc_now_iso()
    trace_catalog["artifacts"] = merge_artifact_ref_dicts(trace_catalog.get("artifacts", []),
                                                          [payload])
    atomic_write_json(trace_path, trace_catalog)

    if audiobook_dir is not None:
        book_path = _audiobook_artifacts_path(audiobook_dir)
        book_catalog = read_json(book_path) or {
            "schema_version": "tts.audiobook_artifacts.v1",
            "audiobook_id": ref.audiobook_id,
            "document_id": ref.document_id,
            "trace_id": ref.trace_id,
            "job_id": ref.job_id,
            "created_at": utc_now_iso(),
            "artifacts": [],
        }
        book_catalog["audiobook_id"] = book_catalog.get("audiobook_id") or ref.audiobook_id
        book_catalog["document_id"] = book_catalog.get("document_id") or ref.document_id
        book_catalog["trace_id"] = book_catalog.get("trace_id") or ref.trace_id
        book_catalog["job_id"] = book_catalog.get("job_id") or ref.job_id
        book_catalog["updated_at"] = utc_now_iso()
        book_catalog["artifacts"] = merge_artifact_ref_dicts(book_catalog.get("artifacts", []),
                                                             [payload])
        atomic_write_json(book_path, book_catalog)

    return ref


def write_document_asset(asset: DocumentAsset, *, cache_dir: Path) -> DocumentAsset:
    payload = model_to_dict(asset)
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(_document_path(cache_dir, asset.document_id), payload)
    return asset


def load_document_asset(document_id: str, *, cache_dir: Path) -> Optional[dict[str, Any]]:
    return read_json(_document_path(cache_dir, document_id))


def load_trace_catalog(trace_id: str, *, cache_dir: Path) -> Optional[dict[str, Any]]:
    return read_json(_catalog_path(cache_dir, trace_id))


def load_audiobook_artifacts(audiobook_dir: Path) -> Optional[dict[str, Any]]:
    return read_json(_audiobook_artifacts_path(audiobook_dir))


def find_artifact_by_role(audiobook_dir: Path, role: str) -> Optional[dict[str, Any]]:
    catalog = load_audiobook_artifacts(audiobook_dir) or {}
    artifacts = catalog.get("artifacts") or []
    candidates = [item for item in artifacts if item.get("role") == role and item.get("path")]
    existing = [item for item in candidates if Path(str(item["path"])).exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda item: str(item.get("created_at") or ""))[-1]


def find_artifact_by_id(artifact_id: str, *, cache_dir: Path, output_dir: Optional[Path] = None) -> \
Optional[dict[str, Any]]:
    for catalog_path in (cache_dir / TRACE_CATALOG_DIRNAME).glob("*.json"):
        catalog = read_json(catalog_path) or {}
        for item in catalog.get("artifacts", []) or []:
            if item.get("artifact_id") == artifact_id:
                return item

    if output_dir and output_dir.exists():
        for catalog_path in output_dir.glob(f"*/{AUDIOBOOK_ARTIFACTS_FILENAME}"):
            catalog = read_json(catalog_path) or {}
            for item in catalog.get("artifacts", []) or []:
                if item.get("artifact_id") == artifact_id:
                    return item
    return None