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
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )

@lru_cache
def get_utility_llm() -> AzureChatOpenAI:
    """A smaller/cheaper model for low-stakes utility tasks (session naming, /btw routing)."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )

@lru_cache
def get_golden_llm() -> AzureChatOpenAI:
    """Model for generating goldens."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_chat_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )
