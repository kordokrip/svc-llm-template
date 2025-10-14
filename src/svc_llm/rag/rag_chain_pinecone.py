from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

from ..llm.chat_model import get_chat
from ..llm.prompts import SYSTEM_BASE, STYLE_KR, FORMAT_DEFAULT
from ..vectorstore.pinecone_store import get_pinecone

# -----------------------------------------------------------------------------
# Pinecone RAG chain (LCEL)
#  - PineconeVectorStore는 내부적으로 OpenAI Embeddings를 사용(프로젝트 설정 기반)
#  - 옵션: MultiQueryRetriever, ContextualCompressionRetriever(LLMChainExtractor)
#  - 히스토리/컨텍스트 안전 주입, 스트리밍 호환(.stream())
# -----------------------------------------------------------------------------


def _join_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _make_prompt(few_shots: Optional[List[Tuple[str, str]]] = None) -> ChatPromptTemplate:
    system = SYSTEM_BASE + "\n\n" + STYLE_KR + "\n\n" + FORMAT_DEFAULT

    msgs: List[Tuple[str, str]] = [("system", system), ("system", "Context:\n{context}"), MessagesPlaceholder("history"), ("human", "{question}")]

    if few_shots:
        example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
        few = FewShotChatMessagePromptTemplate.from_examples(
            examples=[{"input": q, "output": a} for q, a in few_shots],
            example_prompt=example_prompt,
        )
        # system 이후, history 이전에 few-shot 삽입
        msgs = [("system", system), few, ("system", "Context:\n{context}"), MessagesPlaceholder("history"), ("human", "{question}")]

    return ChatPromptTemplate.from_messages(msgs)  # type: ignore[arg-type]


def build_chain_pinecone(
    index_name: str,
    *,
    k: int = 6,
    use_multi_query: bool = True,
    use_compression: bool = True,
    few_shots: Optional[List[Tuple[str, str]]] = None,
):
    """Create a Pinecone-backed RAG LCEL chain.

    Parameters
    ----------
    index_name : str
        Pinecone index name (이미 생성되어 있어야 합니다. 없으면 프로젝트의 get_pinecone가 생성).
    k : int
        top-k for the base retriever.
    use_multi_query : bool
        Enable MultiQueryRetriever over the base retriever. (질의 다각화)
    use_compression : bool
        Enable ContextualCompressionRetriever with LLMChainExtractor. (문맥 압축)
    few_shots : list[tuple[str,str]] | None
        Optional (question, answer) pairs to prepend as few-shot examples.

    Returns
    -------
    Runnable
        LCEL chain supporting .invoke({question, history}) and .stream(...)
    """

    # Vector store & base retriever
    vs = get_pinecone(index_name=index_name)
    base_ret = vs.as_retriever(search_kwargs={"k": k})

    # Optional MultiQuery
    retriever = base_ret
    if use_multi_query:
        llm_for_mq = get_chat(streaming=False)  # 질의 생성은 스트리밍 불필요
        retriever = MultiQueryRetriever.from_llm(retriever=base_ret, llm=llm_for_mq)

    # Optional compression
    if use_compression:
        llm_for_comp = get_chat(streaming=False)
        retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor.from_llm(llm_for_comp),
            base_retriever=retriever,
        )

    prompt = _make_prompt(few_shots)

    # Prepare fields safely: question/history/context
    prepare = RunnableParallel(
        question=lambda x: x["question"],
        history=lambda x: x.get("history", []),
        context=(RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(_join_docs)),
    )

    chain = prepare | prompt | get_chat(streaming=True) | StrOutputParser()
    return chain