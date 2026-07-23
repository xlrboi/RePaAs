"""Session metadata persistence (JSON file) — was inline logic in app.py."""

from __future__ import annotations

import json
import uuid

from folder.config import get_settings
from folder.logging_config import get_logger
from folder.sessions.models import SessionMeta

log = get_logger("sessions.store")


def load_sessions() -> dict[str, dict]:
    settings = get_settings()
    try:
        return json.loads(settings.sessions_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict[str, dict]) -> None:
    settings = get_settings()
    settings.sessions_file.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def create_session() -> SessionMeta:
    meta = SessionMeta(id=str(uuid.uuid4()))
    sessions = load_sessions()
    sessions[meta.id] = meta.model_dump()
    save_sessions(sessions)
    log.info("Created session %s", meta.id)
    return meta
