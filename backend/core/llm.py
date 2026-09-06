"""LLM and embedding model factories.

Every agent obtains its model through this module rather than constructing one
itself, so the model name, temperature and API key are configured in exactly one
place. `get_chat_model` is cached per (model, temperature) pair to avoid
rebuilding clients on every request.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from backend.config import get_settings

logger = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """Raised when the app is started without an LLM API key."""


@lru_cache(maxsize=8)
def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a chat model client.

    Args:
        model: Override the configured model name, e.g. for a cheaper router.
        temperature: Override sampling temperature. Extraction agents use 0.0,
            advisory agents use a slightly higher value for readable prose.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise MissingCredentialsError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return ChatGoogleGenerativeAI(
        model=model or settings.chat_model,
        temperature=settings.temperature if temperature is None else temperature,
        google_api_key=settings.google_api_key,
        max_retries=2,
        timeout=90,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the embedding model used for both ingestion and query time.

    The same model must be used for both, otherwise the query vector and the
    document vectors live in different spaces and retrieval returns noise.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise MissingCredentialsError(
            "GOOGLE_API_KEY is not set, so embeddings cannot be generated."
        )

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
