
from __future__ import annotations
from typing import List, Dict, Tuple

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from .chat_model import get_chat
from .prompts import SYSTEM_BASE

# -----------------------------------------------------------------------------
# Prompt: SYSTEM -> (retrieved) Context -> History -> Human
#  - Context는 별도 system 메시지로 주입해 근거 기반 답변을 유도
#  - History는 ChatMessage 리스트 형태로 안전 변환하여 주입
# -----------------------------------------------------------------------------
BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("system", "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def _format_history(history: List[Dict[str, str]] | List[Tuple[str, str]]):
    """다양한 포맷의 history를 LangChain ChatMessage들로 안전 변환.
    허용 포맷:
      - [{"role":"user|assistant|system", "content": str}, ...]
      - [(role, content), ...]
    """
    msgs = []
    if not history:
        return msgs
    for m in history:
        role, content = (m["role"], m["content"]) if isinstance(m, dict) else (m[0], m[1])
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        elif role == "system":
            msgs.append(SystemMessage(content=content))
    return msgs


def _format_docs(docs) -> str:
    return "\n\n".join(f"- {d.page_content}" for d in docs)


def build_chain(retriever):
    """RAG 체인 빌더 (LCEL).

    입력 형식:
      {"question": str, "history": Optional[List[dict|tuple]]}

    반환:
      - .invoke(payload) -> str
      - .stream(payload) -> Iterable[str] (최종 출력 스트리밍)
    """

    # 1) question/history/context 동시 생성 (입력 보존)
    prepare = RunnableParallel(
        question=lambda x: x["question"],
        history=lambda x: _format_history(x.get("history", [])),
        context=RunnableLambda(lambda x: _format_docs(retriever.get_relevant_documents(x["question"]))),
    )

    # 2) Prompt -> ChatModel -> String
    chain = prepare | BASE_PROMPT | get_chat() | StrOutputParser()
    return chain
