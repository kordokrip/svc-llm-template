from __future__ import annotations
"""
리트리버 유틸리티 (Pydantic v2 호환)
- VectorStore.as_retriever()로 만든 리트리버를 감싸서 MMR/Similarity, Top-K, 질의 확장(어휘 사전)을 지원.
- monkey-patch 금지 → BaseRetriever 정식 서브클래싱으로 래핑.
- .env 기본값을 읽어 안전하게 파라미터 적용.
"""
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import (
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForRetrieverRun,
)

from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

def make_retriever(vs: VectorStore, *, k: int = 6, search_type: str = "similarity") -> BaseRetriever:
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    return vs.as_retriever(search_type=search_type, search_kwargs={"k": k})

# .env 자동 로드
load_dotenv(find_dotenv(), override=False)

# 기본값 (.env에서 없으면 아래 값 사용)
DEFAULT_K = int(os.getenv("SVC_RETRIEVAL_K", "4"))
DEFAULT_SEARCH_TYPE = os.getenv("SVC_RETRIEVAL_SEARCH_TYPE", "similarity").lower()  # or "mmr"
DEFAULT_MMR_LAMBDA = float(os.getenv("SVC_MMR_LAMBDA", "0.5"))
LEXICON_STR = os.getenv("SVC_QUERY_LEXICON", "")  # 예: "스크러버:scrubber|scrubbing; 흡착탑:adsorber|activated"

VALID_SEARCH_TYPES = {"similarity", "mmr"}


def _parse_lexicon(raw: str) -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = {}
    if not raw:
        return table
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        base = k.strip()
        aliases = [x.strip() for x in v.split("|") if x.strip()]
        if base and aliases:
            table[base] = aliases
    return table


def _expand_query(q: str, lexicon: Dict[str, List[str]]) -> str:
    if not q or not lexicon:
        return q
    added: List[str] = []
    q_low = q.lower()
    for base, aliases in lexicon.items():
        if base.lower() in q_low:
            added.extend(aliases)
    if not added:
        return q
    seen = set()
    uniq = [a for a in added if not (a in seen or seen.add(a))]
    return f"{q} " + " ".join(uniq)


def _validate_params(k: int, search_type: str, mmr_lambda: float) -> None:
    if k <= 0:
        raise ValueError(f"k는 1 이상이어야 합니다. (k={k})")
    if search_type not in VALID_SEARCH_TYPES:
        raise ValueError(f"search_type은 {VALID_SEARCH_TYPES} 중 하나여야 합니다. (입력: {search_type})")
    if not (0.0 <= mmr_lambda <= 1.0):
        raise ValueError(f"mmr_lambda는 0.0~1.0 범위여야 합니다. (입력: {mmr_lambda})")


def make_retriever(
    vs: VectorStore,
    k: Optional[int] = None,
    *,
    search_type: Optional[str] = None,
    mmr_lambda: Optional[float] = None,
    lexicon_str: Optional[str] = None,
) -> BaseRetriever:
    # None-safe 기본값
    k = int(DEFAULT_K if k is None else k)
    search_type = (DEFAULT_SEARCH_TYPE if search_type is None else search_type).lower()
    mmr_lambda = float(DEFAULT_MMR_LAMBDA if mmr_lambda is None else mmr_lambda)

    _validate_params(k, search_type, mmr_lambda)

    # 1) 기본 리트리버 생성
    if search_type == "mmr":
        base = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": mmr_lambda})
    else:
        base = vs.as_retriever(search_kwargs={"k": k})

    # 2) 도메인 어휘 사전 기반 질의 확장 래핑
    lex = _parse_lexicon(lexicon_str if lexicon_str is not None else LEXICON_STR)
    if not lex:
        return base

    class LexiconAugmentedRetriever(BaseRetriever):
        """질의 확장 래퍼 (Pydantic v2 호환)"""
        base: BaseRetriever
        lex: Dict[str, List[str]]

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
            **kwargs,
        ) -> List[Document]:
            q2 = _expand_query(query, self.lex)
            cb = run_manager.get_child() if run_manager else None
            return self.base.get_relevant_documents(q2, callbacks=cb, **kwargs)

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
            **kwargs,
        ) -> List[Document]:
            q2 = _expand_query(query, self.lex)
            cb = run_manager.get_child() if run_manager else None
            return await self.base.aget_relevant_documents(q2, callbacks=cb, **kwargs)

    return LexiconAugmentedRetriever(base=base, lex=lex)


if __name__ == "__main__":
    # 간단 자가 테스트
    print("[SELF-TEST] retriever.py 자가 테스트 시작")

    class _DummyRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return [Document(page_content=f"[base] Q={query}")]

    class _DummyVS:
        def as_retriever(self, **kwargs):
            return _DummyRetriever()

    vs = _DummyVS()
    r1 = make_retriever(vs, k=3, search_type="similarity",
                        lexicon_str="스크러버:scrubber|scrubbing; 흡착탑:adsorber")
    docs1 = r1.get_relevant_documents("스크러버 점검 절차")
    print("[TEST] similarity →", docs1[0].page_content)
    assert "scrubber" in docs1[0].page_content and "scrubbing" in docs1[0].page_content

    r2 = make_retriever(vs, k=5, search_type="mmr", mmr_lambda=0.3,
                        lexicon_str="흡착탑:adsorber|activated")
    docs2 = r2.get_relevant_documents("흡착탑 교체 기준")
    print("[TEST] mmr →", docs2[0].page_content)
    assert "adsorber" in docs2[0].page_content

    try:
        _ = make_retriever(vs, k=0)
        raise AssertionError("k=0 이 허용되면 안 됩니다.")
    except ValueError:
        pass
    try:
        _ = make_retriever(vs, search_type="wrong")
        raise AssertionError("잘못된 search_type 이 허용되면 안 됩니다.")
    except ValueError:
        pass
    try:
        _ = make_retriever(vs, mmr_lambda=1.5)
        raise AssertionError("mmr_lambda 범위 초과가 허용되면 안 됩니다.")
    except ValueError:
        pass

    print("[SELF-TEST] OK — 모든 테스트 통과")