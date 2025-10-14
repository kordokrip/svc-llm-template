from __future__ import annotations
# -*- coding: utf-8 -*-
"""Chroma 벡터스토어 생성기 (langchain-chroma, 0.4+ 자동 영속화)."""
import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def _dbg(msg: str) -> None:
    if os.getenv("SVC_DEBUG", "0") not in ("0", "false", "False", ""):
        print(f"[DBG][chroma] {msg}", flush=True)


def _default_embeddings() -> Embeddings:
    # OPENAI_API_KEY 필요
    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    _dbg(f"OpenAIEmbeddings(model={model})")
    return OpenAIEmbeddings(model=model)


def get_chroma(
    *,
    persist_directory: str,
    collection_name: Optional[str] = None,
    embedding_fn: Optional[Embeddings] = None,
):
    coll = collection_name or os.getenv("CHROMA_COLLECTION", "svc-knowledge")
    embedding_fn = embedding_fn or _default_embeddings()
    _dbg(f"Chroma(persist_directory={persist_directory}, collection={coll})")
    vs = Chroma(
        persist_directory=persist_directory,
        collection_name=coll,
        embedding_function=embedding_fn,
    )
    # Chroma 0.4+는 자동 영속화 (persist 호출 불필요)
    return vs