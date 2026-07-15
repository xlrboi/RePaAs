"""tool schemas fortool calling and outputs
"""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel , Field
from folder.constants import *


class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(
        default=DEFAULT_RETRIEVAL_K,
        ge=MIN_RETRIEVAL_K,
        le=MAX_RETRIEVAL_K,
        description="Number of chunks to retrieve",
    )


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(
        default=DEFAULT_WEB_RESULTS,
        ge=MIN_WEB_RESULTS,
        le=MAX_WEB_RESULTS,
        description="Number of web results to return",
    )

