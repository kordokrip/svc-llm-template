
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

# Prefer project settings if available, but fall back to ENV to be robust.
try:
    from ..config import settings  # typing: ignore
except Exception:  # pragma: no cover
    class _Dummy:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")
        pinecone_region = os.getenv("PINECONE_ENV", "us-east-1")
    settings = _Dummy()  # type: ignore

# Try to honor a custom embedding provider if defined, else use OpenAIEmbeddings.
try:
    from ..embeddings.provider import get_embeddings as _get_custom_embeddings  # type: ignore
except Exception:  # pragma: no cover
    _get_custom_embeddings = None


@lru_cache(maxsize=2)
def _embeddings() -> OpenAIEmbeddings:
    if _get_custom_embeddings is not None:
        emb = _get_custom_embeddings()
        if emb is not None:
            return emb
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    dims_env = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
    dims = int(dims_env) if dims_env else None
    return OpenAIEmbeddings(model=model, dimensions=dims)


@lru_cache(maxsize=4)
def get_pinecone(index_name: Optional[str] = None, *, embedding: Optional[OpenAIEmbeddings] = None) -> PineconeVectorStore:
    """Return a ready‑to‑use PineconeVectorStore.

    - Reads API key/index/region from project settings or ENV.
    - Creates a **serverless** index automatically if missing (cosine metric).
    - Embedding defaults to OpenAI `text-embedding-3-small` (dims can be overridden via `OPENAI_EMBEDDING_DIMENSIONS`).
    """
    api_key = getattr(settings, "pinecone_api_key", None) or os.getenv("PINECONE_API_KEY")
    name = index_name or getattr(settings, "pinecone_index", None) or os.getenv("PINECONE_INDEX")
    region = getattr(settings, "pinecone_region", None) or os.getenv("PINECONE_ENV", "us-east-1")
    if not api_key or not name:
        raise RuntimeError("Pinecone 설정 누락(PINECONE_API_KEY / PINECONE_INDEX).")

    pc = Pinecone(api_key=api_key)

    # Check indexes (handle both modern dict response and legacy .names())
    try:
        existing = {i["name"] for i in pc.list_indexes().get("indexes", [])}
    except Exception:  # older client
        try:  # type: ignore[attr-defined]
            existing = set(pc.list_indexes().names())
        except Exception:
            existing = set()

    if name not in existing:
        # Default to 1536 dims (OpenAI text-embedding-3-small). Can be overridden by ENV.
        dims = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS") or 1536)
        pc.create_index(
            name=name,
            dimension=dims,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    index = pc.Index(name)
    emb = embedding or _embeddings()
    return PineconeVectorStore(index=index, embedding=emb)
