"""
Model layer: generator LLM and embeddings.

The generator LLM is consumed through an external API (the main approach of
this work, given the local hardware constraints). Embeddings can run locally
(sentence-transformers, free) or through an API. Everything is configured in
the .env file.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import (
    EMBEDDINGS_MODEL,
    EMBEDDINGS_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


def get_llm() -> BaseChatModel:
    """Return the generator LLM for the configured provider.

    A generous ``max_tokens`` value prevents the RAGAS judge from truncating
    its outputs; only the tokens actually generated are billed.
    """
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS
        )
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def get_embeddings() -> Embeddings:
    """Return the embedding model for the configured provider."""
    if EMBEDDINGS_PROVIDER == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    if EMBEDDINGS_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {EMBEDDINGS_PROVIDER}")
