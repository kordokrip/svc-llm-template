from __future__ import annotations
import os
import argparse
from typing import List, Optional, Tuple

from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

# (선택) MultiQuery, Ensemble 사용 시 임포트 — 함수 내부에서 재임포트하지 마세요!
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever  # type: ignore
except Exception:  # 패키지 미설치/버전 차이 등
    MultiQueryRetriever = None  # type: ignore

try:
    from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore
except Exception:
    EnsembleRetriever = None  # type: ignore


def build_pinecone_retriever(
    index_name: str,
    namespace: Optional[str],
    k: int,
    search_type: str,
    embed_model: str,
):
    """Pinecone 인덱스에서 리트리버 생성 (LangChain 0.3 스타일).

    - from_existing_index API는 기존 인덱스/네임스페이스를 안전하게 재사용합니다.
    - search_type: "similarity" | "mmr"
    """
    embed = OpenAIEmbeddings(model=embed_model)
    vs = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        namespace=namespace,
        embedding=embed,
    )
    if search_type == "mmr":
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.5})
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def build_chain_pinecone(
    chat_model: str,
    retriever,
    few_shots: Optional[List[Tuple[str, str]]] = None,
    stream: bool = False,
):
    """Retrieval → Prompt → LLM 체인 구성.
    컨텍스트 포맷터에서 source 메타데이터를 함께 출력하도록 구성합니다.
    """
    llm = ChatOpenAI(model=chat_model, temperature=0.2, streaming=stream)

    sys_msg = (
        "당신은 SVC 도메인 문서를 바탕으로 정확하고 간결하게 답하는 어시스턴트입니다. "
        "출처가 되는 문서의 핵심을 근거로 답하세요. 모르면 모른다고 말하세요."
    )
    messages = [
        ("system", sys_msg),
        (
            "human",
            "질문: {question}\n\n[검색 컨텍스트]\n{context}\n\n요청: 위 컨텍스트를 우선 근거로 한국어로 답하세요.",
        ),
    ]

    if few_shots:
        for q, a in few_shots:
            messages.insert(1, ("human", f"예시 질문: {q}"))
            messages.insert(2, ("ai", a))

    prompt = ChatPromptTemplate.from_messages(messages)

    def format_docs(docs: List[Document]) -> str:
        out = []
        for d in docs:
            src = d.metadata.get("source", "")
            page = d.metadata.get("page")
            row = d.metadata.get("row")
            loc = f" p{page}" if page is not None else (f" row{row}" if row is not None else "")
            out.append(f"[source:{src}{loc}] {d.page_content}")
        return "\n\n".join(out)

    setup = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    chain = setup | prompt | llm
    return chain


def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.getenv("PINECONE_INDEX", "svc-knowledge"))
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--search-type", choices=["similarity", "mmr"], default="similarity")
    ap.add_argument("--multiquery", action="store_true", help="LLM으로 질의 확장(MultiQueryRetriever)")
    ap.add_argument("--ensemble", action="store_true", help="리트리버 앙상블(similarity+mmr)")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--embed-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    ap.add_argument("--chat-model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    ap.add_argument("--question", default="CH1Idle상태일때GPM하강시 조치 방법 알려줘.")
    ap.add_argument("--show-sources", action="store_true", help="Retrieval 결과 소스 경로를 함께 출력")
    args = ap.parse_args()

    base = build_pinecone_retriever(
        index_name=args.index,
        namespace=args.namespace,
        k=args.k,
        search_type=args.search_type,
        embed_model=args.embed_model,
    )

    retriever = base

    # (선택) MultiQuery: 질의 다양화로 리콜 향상
    if args.multiquery and MultiQueryRetriever is not None:
        mq_llm = ChatOpenAI(model=args.chat_model, temperature=0.3)
        retriever = MultiQueryRetriever.from_llm(retriever=base, llm=mq_llm)

    # (선택) Ensemble: MMR+similarity 가중합(랭크퓨전)
    if args.ensemble and EnsembleRetriever is not None:
        mmr_ret = build_pinecone_retriever(
            index_name=args.index,
            namespace=args.namespace,
            k=args.k,
            search_type="mmr",
            embed_model=args.embed_model,
        )
        retriever = EnsembleRetriever(retrievers=[base, mmr_ret], weights=[0.5, 0.5])

    chain = build_chain_pinecone(
        chat_model=args.chat_model,
        retriever=retriever,
        stream=args.stream,
    )

    print(
        f"[INFO] Pinecone index='{args.index}', namespace='{args.namespace}', k={args.k}, "
        f"search={args.search_type}, multiquery={bool(args.multiquery)}, ensemble={bool(args.ensemble)}"
    )
    print("[Q]", args.question)

    ans = chain.invoke(args.question)
    print("\n[ANSWER]\n", ans.content)

    if args.show_sources:
        try:
            docs: List[Document] = retriever.invoke(args.question)  # type: ignore
            print("\n[SOURCES]")
            for i, d in enumerate(docs[: args.k], 1):
                src = d.metadata.get("source", "")
                page = d.metadata.get("page")
                row = d.metadata.get("row")
                loc = f" p{page}" if page is not None else (f" row{row}" if row is not None else "")
                print(f"{i}. {src}{loc}")
        except Exception as e:
            print("[SOURCES] 출력 중 오류:", repr(e))


if __name__ == "__main__":
    main()