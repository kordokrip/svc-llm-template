from __future__ import annotations
# -*- coding: utf-8 -*-
"""
rag_chain.py
- LangChain 0.3.x API에 맞춘 RAG 체인 빌더
- FewShotChatMessagePromptTemplate: from_examples 없음 → 생성자에 examples 전달
- MultiQueryRetriever / ContextualCompressionRetriever 최신 사용법 적용
- 최종 출력은 문자열이 되도록 StrOutputParser 연결
"""

import os
from operator import itemgetter
from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv
# VS Code/REPL/헤리독에서도 안전하게 .env 찾기
load_dotenv(find_dotenv(usecwd=True), override=False)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from ..vectorstore.chroma_store import get_chroma
from .retriever import make_retriever


# ---------- helpers ----------

def _format_docs(docs) -> str:
    """검색된 문서 리스트를 모델 입력용 문자열로 합칩니다."""
    return "\n\n".join(d.page_content for d in docs)


def _fewshot_prompt(few_shots: List[Tuple[str, str]] | None):
    """FewShotChatMessagePromptTemplate 생성 (0.3: 생성자 사용)"""
    if not few_shots:
        return None
    examples = [{"input": q, "output": a} for q, a in few_shots]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    return FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )


def _make_prompt(few_shots: List[Tuple[str, str]] | None) -> ChatPromptTemplate:
    """대화 프롬프트 구성(시스템+few-shot+히스토리+사용자 질문)"""
    sys_msg = (
        "당신은 SVC 내부 문서를 바탕으로 정확하고 간결하게 답하는 어시스턴트입니다. "
        "출처가 불명확하면 모른다고 답하고, 제공 문맥을 우선합니다."
    )
    msgs = [("system", sys_msg)]

    fs = _fewshot_prompt(few_shots)
    if fs is not None:
        msgs.append(fs)

    msgs += [
        MessagesPlaceholder("history"),
        ("human",
         "질문: {question}\n\n[검색 문맥]\n{context}\n\n지침:\n- 문맥 내에서만 답변\n- 핵심 bullet로 요약\n- 필요한 경우 표기준/항목을 나열"),
    ]
    return ChatPromptTemplate.from_messages(msgs)


# ---------- main builder ----------

def build_chain(
    *,
    persist_directory: str,
    collection_name: str | None = None,
    k: int = 6,
    use_multi_query: bool = True,
    use_compression: bool = True,
    few_shots: List[Tuple[str, str]] | None = None,
) -> Runnable:
    """RAG 체인 구성 후 Runnable 반환 (invoke/stream 지원)

    - 로컬 Chroma 벡터스토어 로드, OpenAIEmbeddings 사용
    - Retriever는 as_retriever로 생성(몽키패치 금지)
    - 옵션: MultiQuery / Contextual Compression
    - 최종 출력: 문자열
    """

    # 1) 임베딩 & 벡터스토어
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embeddings = OpenAIEmbeddings(model=embed_model)

    vs = get_chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_fn=embeddings,  # 중요: 반드시 전달
    )

    # 2) 기본 retriever
    base_ret = make_retriever(vs, k=k)

    # 3) (선택) 쿼리 다양화
    if use_multi_query:
        mq_llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
        base_ret = MultiQueryRetriever.from_llm(
            retriever=base_ret,
            llm=mq_llm,
            include_original=True,
        )

    # 4) (선택) 문맥 압축으로 잡음 제거
    if use_compression:
        comp_llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
        compressor = LLMChainExtractor.from_llm(comp_llm)
        retriever = ContextualCompressionRetriever(
            base_retriever=base_ret,
            base_compressor=compressor,
        )
    else:
        retriever = base_ret

    # 5) LLM + 프롬프트
    llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    prompt = _make_prompt(few_shots)

    # 6) LCEL: {'context','question','history'} -> prompt -> llm -> str
    chain: Runnable = (
        {
            "context": itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
        | StrOutputParser()  # stream 시 str 청크가 나옵니다
    )
    return chain