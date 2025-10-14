from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 
load_dotenv()

_DEF_PERSIST = os.getenv("CHROMA_PERSIST", ".chroma")
_DEF_COLLECTION = os.getenv("CHROMA_COLLECTION", "svc_collection")

# Embedding ENV
_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_EMBED_DIMS = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")


@lru_cache(maxsize=2)
def _get_embeddings() -> OpenAIEmbeddings:
    # OpenAI text-embedding-3-small defaults to 1536 dims; can be reduced via dimensions param.
    # Ref: OpenAI embeddings guide
    dims = int(_EMBED_DIMS) if _EMBED_DIMS else None
    return OpenAIEmbeddings(model=_EMBED_MODEL, dimensions=dims)


@lru_cache(maxsize=4)
def get_chroma(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    *,
    embedding: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """Return a persisted Chroma vector store.

    If the directory does not exist, it will be created on first `add_*`.
    """
    persist_directory = persist_directory or _DEF_PERSIST
    collection_name = collection_name or _DEF_COLLECTION

    emb = embedding or _get_embeddings()
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=emb,
    )
    return vs
