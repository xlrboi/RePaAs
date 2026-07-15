"""Single factory for llm instances.
"""

from __future__ import annotations
from functools import lru_cache
from langchain_openai import AzureChatOpenAI
from folder.config import get_settings


@lru_cache
def get_main_llm() -> AzureChatOpenAI:
    """The reasoning/generation model used throughout the RAG graph."""
    settings = get_settings()
    return AzureChatOpenAI(azure_deployment=settings.azure_openai_chat_deployment, azure_api_key=settings.azure_openai_api_key)

