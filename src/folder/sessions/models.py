"""Session metadata model"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    id: str
    name: str = "New Session"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_named: bool = False
