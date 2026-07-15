"""TavilyClient wrapper 
"""

from __future__ import annotations
from functools import lru_cache
from typing import Any
from tavily import TavilyClient
from folder.config import get_settings


@lru_cache
def _get_client() -> TavilyClient:
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


def search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run a Tavily web search and return the raw ``results`` list."""
    response = _get_client().search(query, max_results=max_results)
    return response.get("results", [])
