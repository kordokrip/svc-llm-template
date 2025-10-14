from __future__ import annotations
# -*- coding: utf-8 -*-
"""
scripts/test_rag_chain_pinecone.py
- LangChain 0.3.x 스타일의 RAG 체인 테스트 스크립트 (Pinecone 버전)
- .env 로딩 → 체인(build_chain) 생성 → 스트리밍/일반 실행 지원
- 필요 시, 동일 파라미터로 top-k 소스 미리보기도 제공(--show-sources)

실행 예시
----------
# 스트리밍 응답
PYTHONPATH=$(pwd)/src \
python scripts/test_rag_chain_pinecone.py \
  --index svc-knowledge --namespace svc-v1 --stream \
  --question "흡착탑 교체 기준 요약"

# 일반 응답 + 소스 확인
PYTHONPATH=$(pwd)/src \
python scripts/test_rag_chain_pinecone.py \
  --index svc-knowledge --namespace svc-v1 --k 6 --show-sources \
  --question "스크러버 점검 절차"
"""

import argparse
import os
import sys
from typing import List, Tuple

# Pinecone SDK는 선택사항. 인덱스 사전검증에만 사용.
try:
    from pinecone import Pinecone  # SDK v4.x
except Exception:
    Pinecone = None  # SDK 미설치 시에도 스크립트 동작하게 방어

from dotenv import load_dotenv, find_dotenv

# ---- 안전한 .env 로딩 (VSCode/REPL 고려) ----
load_dotenv(find_dotenv(usecwd=True), override=False)

# 체인 빌더 불러오기 (우리 프로젝트 모듈)
try:
    sys.path.append(os.path.abspath("src"))
    from svc_llm.rag.rag_chain_pinecone import build_chain
    from svc_llm.vectorstore.pinecone_store import get_pinecone
    from svc_llm.rag.retriever import make_retriever
    from langchain_openai import OpenAIEmbeddings
except Exception as e:
    print("[IMPORT-ERROR] PYTHONPATH 설정을 확인하세요. e=", e)
    print("예: PYTHONPATH=$(pwd)/src python scripts/test_rag_chain_pinecone.py --index svc-knowledge --namespace svc-v1")
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 체인 테스트 (Pinecone)")
    parser.add_argument("--index", default=os.getenv("PINECONE_INDEX", ""), help="Pinecone 인덱스명 (미지정 시 PINECONE_INDEX 환경변수 사용)")
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", ""), help="Pinecone 네임스페이스")
    parser.add_argument("--k", type=int, default=6, help="retriever top-k")
    parser.add_argument("--multiquery", action="store_true", help="MultiQueryRetriever 활성화")
    parser.add_argument("--ensemble", action="store_true", help="EnsembleRetriever(랭크 융합) 활성화")
    parser.add_argument("--compress", action="store_true", help="ContextualCompression 활성화")
    parser.add_argument("--stream", action="store_true", help="LLM 스트리밍 출력")
    parser.add_argument("--question", default="현장 안전 점검 핵심 체크리스트 요약", help="질문 프롬프트")
    parser.add_argument("--show-sources", action="store_true", help="쿼리로 검색된 상위 문서 메타데이터 미리보기")
    return parser.parse_args()


def _resolve_index_name(idx: str | None) -> str:
    """CLI 인자 또는 환경변수에서 Pinecone 인덱스명을 확정한다."""
    name = (idx or os.getenv("PINECONE_INDEX") or "").strip()
    if not name:
        raise SystemExit("[ERROR] Pinecone 인덱스명이 비어있습니다. --index 옵션을 지정하거나 PINECONE_INDEX 환경변수를 설정하세요.")
    return name


def preview_sources(query: str, *, index: str, namespace: str, k: int) -> None:
    """동일 임베딩/벡터스토어로 top-k 소스 미리보기(디버깅용)."""
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=embed_model)
    vs = get_pinecone(index_name=index, namespace=namespace, embedding_fn=embeddings)
    retr = make_retriever(vs, k=k)
    docs = retr.invoke(query)
    print("\n[SOURCES] 상위 문서 미리보기 (k=%d)" % k)
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("path") or meta.get("id") or "(unknown)"
        print(f" {i:>2}. {src}")


def default_few_shots() -> List[Tuple[str, str]]:
    """간단한 few-shot 예시 (필요 시 수정)."""
    return [
        ("SOP 문서에서 핵심만 뽑아 bullet로 정리해줘.", "• 개요\n• 목적\n• 절차 요약\n• 점검/주의사항"),
        ("유지보수 교체 기준을 정리해줘.", "• 교체 주기\n• 성능 저하 기준\n• 안전/품질 임계치\n• 기록 및 보고"),
    ]


def main() -> None:
    args = parse_args()

    # 인덱스명 확정 및 사전 검증
    args.index = _resolve_index_name(args.index)
    if Pinecone is not None and os.getenv("PINECONE_API_KEY"):
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            # Pinecone SDK 4.x: list_indexes() → Index 목록
            listed = pc.list_indexes()
            existing = []
            for it in listed:
                name = getattr(it, "name", None)
                if name is None and isinstance(it, dict):
                    name = it.get("name")
                if name is None and isinstance(it, str):
                    name = it
                if name:
                    existing.append(name)
            if existing and args.index not in existing:
                raise SystemExit(f"[ERROR] Index '{args.index}' 가 프로젝트에 없습니다. 기존 인덱스: {sorted(existing)}")
        except Exception as e:
            print(f"[WARN] Pinecone 인덱스 존재 여부 확인을 건너뜁니다: {e}")
    else:
        print("[WARN] Pinecone SDK 또는 PINECONE_API_KEY 가 없어 인덱스 사전검증을 생략합니다.")

    print(f"[INFO] index={args.index}, namespace={args.namespace or '(default)'}, k={args.k}")
    print(f"[INFO] use_multi_query={args.multiquery}, use_ensemble={args.ensemble}, use_compression={args.compress}")

    chain = build_chain(
        index_name=args.index,
        namespace=args.namespace,
        k=args.k,
        use_multi_query=args.multiquery,
        use_ensemble=args.ensemble,
        use_compression=args.compress,
        few_shots=default_few_shots(),
    )

    q = args.question
    if args.show_sources:
        preview_sources(q, index=args.index, namespace=args.namespace, k=args.k)

    print("\n[Q]", q)
    if args.stream:
        print("[MODE] stream")
        for chunk in chain.stream({"question": q, "history": []}):
            print(chunk, end="", flush=True)
        print()
    else:
        print("[MODE] invoke")
        ans = chain.invoke({"question": q, "history": []})
        print(ans)


if __name__ == "__main__":
    main()