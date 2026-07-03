# ~/TTS/my_app/service_health.py

"""Shared health response helpers for the local TTS service family."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def check_ok(value: Any) -> bool:
    """Return True when a health-check value represents an operational state."""
    if value is True:
        return True
    if value is False or value is None:
        return False
    text = str(value).strip().lower()
    return not (
            text == ""
            or text == "false"
            or text.startswith("error")
            or text.startswith("missing")
            or text.startswith("unavailable")
    )


def build_health_response(
        *,
        service: str,
        role: str,
        checks: Mapping[str, Any] | None = None,
        version: str = "0.1.0",
        details: Mapping[str, Any] | None = None,
        status: str | None = None,
) -> dict[str, Any]:
    """Build the stable health envelope shared by all TTS app services."""
    normalized_checks = dict(checks or {})
    resolved_status = status or (
        "ok" if all(check_ok(value) for value in normalized_checks.values()) else "degraded"
    )
    return {
        "status": resolved_status,
        "service": service,
        "role": role,
        "version": version,
        "checks": normalized_checks,
        "details": dict(details or {}),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }