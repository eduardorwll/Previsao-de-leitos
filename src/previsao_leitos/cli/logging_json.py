"""Emissor de logs estruturados em JSON para rastreabilidade reprodutível (Feature F07)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime


def resolve_git_sha() -> str:
    """Resolve SHA curto do commit Git atual sem lançar erro fora de repositórios."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def format_json_log_record(
    *,
    run_id: str,
    entry_point: str,
    event: str,
    payload: Mapping[str, object] | None = None,
    level: str = "INFO",
) -> str:
    """Serializa evento estruturado em JSON (jamais deve incluir credenciais ou PII)."""
    record: dict[str, object] = {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "level": level,
        "run_id": run_id,
        "entry_point": entry_point,
        "git_sha": resolve_git_sha(),
        "event": event,
        "payload": dict(payload or {}),
    }
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def emit_json_log(
    *,
    run_id: str,
    entry_point: str,
    event: str,
    payload: Mapping[str, object] | None = None,
    level: str = "INFO",
) -> None:
    """Escreve linha JSON estruturada em stdout."""
    line = format_json_log_record(
        run_id=run_id,
        entry_point=entry_point,
        event=event,
        payload=payload,
        level=level,
    )
    print(line, file=sys.stdout)
