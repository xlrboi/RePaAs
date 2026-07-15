"""Structured-output schemas used during the retrieval agent's relevancy check."""

from __future__ import annotations
from pydantic import BaseModel


class RelevancyDecision(BaseModel):
    is_relevant: bool
    reason: str
