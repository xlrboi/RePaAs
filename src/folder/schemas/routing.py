"""Structured-output schemas for query routing decisions."""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class RouterDecision(BaseModel):
    """Main graph router: classifies a user query into one workflow branch. Works on the router node where llm call will be made to route decision based on the query."""

    route: Literal["retrieve", "verify_claim", "direct_answer"]


class BtwRouteDecision(BaseModel):
    """Side-channel ``/btw`` router: web search vs. direct knowledge answer."""

    needs_web_search: bool
