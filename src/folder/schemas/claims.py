"""Structured-output schemas for the claim-verification workflow."""

from __future__ import annotations
from pydantic import BaseModel


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]
