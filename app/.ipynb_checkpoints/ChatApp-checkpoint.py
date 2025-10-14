import html
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from svc_llm.llm.chat_model import get_chat
from svc_llm.rag.rag_chain_pinecone import build_chain_pinecone
from svc_llm.vectorstore.pinecone_store import get_pinecone

load_dotenv()

st.set_page_config(page_title="GST CS AI ChatBot", page_icon="💬", layout="wide")
st.title("")
st.markdown(
    """
    <style>
    body, .stApp {
        font-family: 'Plus Jakarta Sans', 'Pretendard', sans-serif;
        background: radial-gradient(circle at 14% 20%, rgba(196,226,255,0.45), transparent 52%),
                    radial-gradient(circle at 80% 10%, rgba(242,210,255,0.45), transparent 55%),
                    linear-gradient(135deg, #f9fbff 0%, #f5f0ff 45%, #fef8f4 100%);
        color: #102036;
    }
    div.block-container {
        padding: 0 2.6rem 3rem 2.6rem;
        max-width: 1250px;
        margin: 0 auto;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(18px);
        color: #1e293b;
        padding-top: 1.5rem;
        border-right: 1px solid rgba(15,23,42,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #2b3445 !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stSlider [role="slider"] {
        background: linear-gradient(135deg, #38bdf8, #a855f7);
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #475569 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
    section[data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.7);
        border-radius: 14px;
        border: 1px solid rgba(100,116,139,0.18);
    }
    section[data-testid="stSidebar"] .stToggle {
        background: rgba(241,245,249,0.7);
        border-radius: 20px;
        padding: 0.45rem 0.6rem;
        border: 1px solid rgba(148,163,184,0.25);
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #a855f7);
        border: none;
        color: #fff !important;
        border-radius: 14px;
        font-weight: 600;
        box-shadow: 0 14px 28px rgba(168,85,247,0.25);
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0 0.6rem 1.6rem 0.6rem;
    }
    .sidebar-chip-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(59,130,246,0.85), rgba(236,72,153,0.8));
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 3px 8px rgba(255,255,255,0.35), 0 16px 28px rgba(59,130,246,0.25);
    }
    .sidebar-chip-icon svg {
        width: 24px;
        height: 24px;
        fill: #f8fafc;
    }
    .sidebar-title {
        margin: 0;
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .sidebar-subtitle {
        margin: 0;
        font-size: 0.78rem;
        opacity: 0.65;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .icon-stack {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        margin: 1rem 0 1.6rem 0;
    }
    .icon-toolbar .stButton > button {
        width: 56px;
        height: 56px;
        border-radius: 20px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 16px 28px rgba(148,163,184,0.2);
        font-size: 1.15rem;
        cursor: pointer;
    }
    .icon-toolbar .stButton > button:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.95);
        box-shadow: 0 20px 30px rgba(99,102,241,0.22);
    }
    .icon-toolbar .stButton {
        margin-bottom: 0.5rem;
    }
    .icon-panel {
        margin-top: 1.2rem;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 16px 32px rgba(148,163,184,0.18);
        backdrop-filter: blur(12px);
    }
    .icon-panel h4 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        color: #0f172a;
    }
    .icon-chip {
        width: 48px;
        height: 48px;
        border-radius: 18px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 12px 18px rgba(148,163,184,0.18);
        font-size: 1.12rem;
    }
    .conversation-header {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }
    .conversation-item {
        padding: 0.85rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 16px 32px rgba(148,163,184,0.18);
        font-size: 0.88rem;
        line-height: 1.4;
        color: #182230;
        backdrop-filter: blur(12px);
    }
    .conversation-item + .conversation-item {
        margin-top: 0.65rem;
    }
    .conversation-empty {
        padding: 1.25rem;
        border-radius: 18px;
        background: rgba(248,250,252,0.75);
        border: 1px dashed rgba(148,163,184,0.4);
        color: #64748b;
        font-size: 0.9rem;
    }
    .chat-hero {
        position: relative;
        overflow: hidden;
        border-radius: 26px;
        padding: 1.9rem 2.1rem;
        background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(221,241,255,0.85), rgba(255,232,250,0.85));
        color: #111827;
        box-shadow: 0 32px 60px rgba(148,163,184,0.28);
        margin-bottom: 1.3rem;
        border: 1px solid rgba(148,163,184,0.18);
        backdrop-filter: blur(16px);
    }
    .chat-hero::after {
        content: "";
        position: absolute;
        inset: -35% 50% auto -20%;
        height: 260px;
        background: radial-gradient(circle, rgba(129,212,250,0.45) 0%, transparent 70%);
        pointer-events: none;
    }
    .chat-hero__wrap {
        position: relative;
        display: flex;
        gap: 1.7rem;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        z-index: 1;
    }
    .chat-hero__icon {
        width: 72px;
        height: 72px;
        border-radius: 22px;
        background: linear-gradient(140deg, rgba(59,130,246,0.85), rgba(236,72,153,0.8));
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 3px 10px rgba(255,255,255,0.35), 0 20px 36px rgba(99,102,241,0.25);
    }
    .chat-hero__icon svg {
        width: 56px;
        height: 56px;
        fill: #f8fafc;
    }
    .chat-hero__body h2 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: #0f172a;
    }
    .chat-hero__body {
        flex: 1 1 250px;
    }
    .chat-hero__body p {
        margin: 0.52rem 0 0 0;
        font-size: 0.98rem;
        color: #334155;
        line-height: 1.6;
    }
    .chat-hero__meta {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        font-size: 0.82rem;
        color: #475569;
        align-items: flex-end;
        min-width: 170px;
        text-align: right;
    }
    .chip {
        border-radius: 999px;
        padding: 0.32rem 0.95rem;
        background: rgba(248,250,255,0.9);
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid rgba(148,163,184,0.22);
        backdrop-filter: blur(10px);
    }
    .chip + .chip {
        margin-left: 0.45rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 22px;
        padding: 1.1rem 1.3rem;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 20px 36px rgba(148,163,184,0.22);
        margin-bottom: 1rem;
        backdrop-filter: blur(14px);
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: rgba(240,249,255,0.92);
    }
    [data-testid="stChatMessage"] pre, [data-testid="stChatMessage"] p {
        font-size: 1.02rem;
        line-height: 1.65;
        color: #0f172a;
    }
    .src-pill {
        display: inline-block;
        border: 1px solid rgba(56,189,248,0.35);
        border-radius: 999px;
        padding: 0.22rem 0.65rem;
        margin: 0.22rem 0.35rem 0.22rem 0;
        font-size: 0.76rem;
        background: rgba(56,189,248,0.12);
        color: #0f4a6a;
        font-weight: 600;
    }
    .feedback-box {
        margin-top: 1.6rem;
        padding: 1.15rem 1.3rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 18px 36px rgba(148,163,184,0.18);
    }
    .feedback-box h4 {
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #1f2937;
    }
    .feedback-box button {
        width: 100%;
        border-radius: 18px !important;
        padding: 0.45rem 0 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 1.1rem;
        align-items: center;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(255,255,255,0.9);
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 22px 36px rgba(148,163,184,0.22);
        padding: 0.4rem 0.6rem;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1.02rem;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CHIP_ICON = """
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad-base" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1f5eff"/>
      <stop offset="100%" stop-color="#2dd4ff"/>
    </linearGradient>
    <linearGradient id="grad-wafer" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#e0f2ff"/>
      <stop offset="100%" stop-color="#fef9c3"/>
    </linearGradient>
    <linearGradient id="grad-stand" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <radialGradient id="grad-shadow" cx="50%" cy="40%" r="65%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#3b82f6" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="18" y="76" width="84" height="22" rx="10" fill="url(#grad-base)" opacity="0.9"/>
  <g opacity="0.9">
    <path d="M23 68h14v18h-14zM83 68h14v18h-14zM53 68h14v18h-14z" fill="url(#grad-stand)"/>
  </g>
  <circle cx="60" cy="58" r="40" fill="url(#grad-wafer)" stroke="#60a5fa" stroke-width="4"/>
  <circle cx="60" cy="54" r="30" fill="url(#grad-shadow)" opacity="0.55"/>
  <g transform="translate(34 33)">
    <g fill="#fde047">
      <rect x="0" y="0" width="10" height="10" rx="2"/>
      <rect x="10" y="10" width="10" height="10" rx="2" fill="#38bdf8"/>
      <rect x="20" y="0" width="10" height="10" rx="2" fill="#f97316"/>
      <rect x="30" y="10" width="10" height="10" rx="2" fill="#22c55e"/>
      <rect x="0" y="20" width="10" height="10" rx="2" fill="#f9a8d4"/>
      <rect x="20" y="20" width="10" height="10" rx="2" fill="#34d399"/>
      <rect x="10" y="0" width="10" height="10" rx="2" fill="#60a5fa"/>
      <rect x="30" y="0" width="10" height="10" rx="2" fill="#c084fc"/>
      <rect x="0" y="10" width="10" height="10" rx="2" fill="#fb7185"/>
    </g>
  </g>
  <g fill="#38bdf8" opacity="0.92">
    <path d="M44 20h8v12h-8z"/>
    <path d="M68 20h8v12h-8z"/>
    <path d="M54 12h12v8H54z"/>
  </g>
  <path d="M50 34h20l-10 14z" fill="#2563eb" opacity="0.85"/>
</svg>
"""

ICON_ACTIONS = [
    {"icon": "＋", "label": "새 대화", "action": "new_chat"},
    {"icon": "🧭", "label": "추천 프롬프트", "action": "recommend"},
    {"icon": "⭐", "label": "즐겨찾기", "action": "favorites"},
    {"icon": "🗓️", "label": "일정", "action": "schedule"},
    {"icon": "📎", "label": "파일", "action": "files"},
    {"icon": "⚙️", "label": "설정", "action": "settings"},
]
PROMPT_SUGGESTIONS = [
    "SOP | 챔버 온도 이탈 시 1차 대응 프로세스는?",
    "알람 | Heater TS EG0-60-07 발생 원인과 조치 순서?",
    "자재 | EBR KIT 교체 주기와 재고 확보 체크리스트?",
    "정산 | 최근 3개월 미해결 CS건 요약 및 비용 추산?",
    "보고 | 주간 품질 리포트에 포함해야 할 핵심 KPI는?",
]


def _init_states() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages: List[Tuple[str, str]] = []
    if "_rag_cache" not in st.session_state:
        st.session_state._rag_cache: Dict[str, object] = {}
    if "feedback_stats" not in st.session_state:
        st.session_state.feedback_stats = {"like": 0, "bad": 0}
    if "feedback_used" not in st.session_state:
        st.session_state.feedback_used: Set[int] = set()
    if "timeline" not in st.session_state:
        st.session_state.timeline: List[Dict[str, str]] = []
    if "last_response_id" not in st.session_state:
        st.session_state.last_response_id = None
    if "last_answer_text" not in st.session_state:
        st.session_state.last_answer_text = ""
    if "icon_modal" not in st.session_state:
        st.session_state.icon_modal = None
    if "icon_alert" not in st.session_state:
        st.session_state.icon_alert = None


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.timeline = []
    st.session_state.last_response_id = None
    st.session_state.last_answer_text = ""
    st.session_state.icon_modal = None
    st.session_state.icon_alert = None


def _handle_icon_action(action: str) -> None:
    if action == "new_chat":
        _reset_conversation()
        st.session_state.icon_alert = ("success", "새 대화가 시작되었습니다. 오른쪽 입력창에서 질문을 시작하세요.")
        return
    st.session_state.icon_modal = action


def _close_icon_modal() -> None:
    st.session_state.icon_modal = None


def _append_message(role: str, content: str) -> None:
    st.session_state.messages.append((role, content))
    st.session_state.timeline.append(
        {
            "role": role,
            "content": content,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(st.session_state.timeline) > 60:
        del st.session_state.timeline[: len(st.session_state.timeline) - 60]


def _history_payload(messages: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    payload: List[Dict[str, str]] = []
    for role, content in messages:
        if role in {"user", "assistant", "system"}:
            payload.append({"role": role, "content": content})
    return payload


def _extract_text(chunk: object) -> str:
    if isinstance(chunk, (AIMessage, AIMessageChunk)):
        return chunk.content or ""
    if isinstance(chunk, BaseMessage):
        return chunk.content or ""
    return str(chunk or "")


def _get_rag_components(
    index_name: str,
    k: int,
    use_multi_query: bool,
    use_compression: bool,
):
    cache: Dict[str, object] = st.session_state._rag_cache  # type: ignore[attr-defined]
    key = (index_name, k, use_multi_query, use_compression)
    if cache.get("key") == key:
        return cache["chain"], cache["retriever"]  # type: ignore[return-value]

    vs = get_pinecone(index_name=index_name)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    if use_multi_query:
        try:
            from langchain.retrievers.multi_query import MultiQueryRetriever

            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=get_chat(streaming=False))
        except Exception:
            st.warning("MultiQueryRetriever 초기화에 실패하여 기본 검색으로 진행합니다.", icon="⚠️")

    if use_compression:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor

            compressor = LLMChainExtractor.from_llm(get_chat(streaming=False))
            retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
        except Exception:
            st.warning("문맥 압축 초기화에 실패하여 압축 없이 진행합니다.", icon="⚠️")

    chain = build_chain_pinecone(
        index_name=index_name,
        k=k,
        use_multi_query=use_multi_query,
        use_compression=use_compression,
        few_shots=None,
    )

    cache["key"] = key
    cache["chain"] = chain
    cache["retriever"] = retriever
    return chain, retriever


def _render_sources(docs: List[Document]) -> str:
    pills = []
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("source") or meta.get("file_path") or f"Doc{idx}"
        score = meta.get("score")
        suffix = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""
        pills.append(f'<span class="src-pill">{html.escape(title)}{suffix}</span>')
    return "".join(pills)


def _render_sidebar() -> Dict[str, object]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <div class="sidebar-chip-icon">{CHIP_ICON}</div>
                <div>
                    <p class="sidebar-title">GST CS AI ChatBot</p>
                    <p class="sidebar-subtitle">Semiconductor Service</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("⚙️ 대화 설정")
        model_default = os.getenv("MODEL_FT") or os.getenv("SVC_MODEL") or "ft:gpt-4.1-mini-2025-04-14:personal:svc-41mini-sft-dpo-80usd-sft:CLB4qudK"
        model_options = list(dict.fromkeys([
            model_default,
            "ft:gpt-4.1-mini-2025-04-14:personal:svc-41mini-sft-dpo-80usd-sft:CLB4qudK",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4o-mini-2024-11-20",
        ]))
        model = st.selectbox("모델", model_options, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("검색 Top-K", 1, 10, 4, 1)
        use_rag = st.toggle("RAG 사용", True)
        use_multi_query = st.toggle("Multi Query 확장", True)
        use_compression = st.toggle("문맥 압축", True)
        show_sources = st.toggle("근거 문서 표시", True)
        pinecone_index = st.text_input("Pinecone Index", os.getenv("PINECONE_INDEX", ""))
        if st.button("대화 초기화", use_container_width=True):
            _reset_conversation()
            st.success("대화 히스토리를 초기화했습니다.")
        return {
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "use_rag": use_rag,
            "use_multi_query": use_multi_query,
            "use_compression": use_compression,
            "show_sources": show_sources,
            "pinecone_index": pinecone_index.strip(),
        }


def _render_history_panel() -> None:
    toolbar = st.container()
    with toolbar:
        st.markdown('<div class="icon-toolbar">', unsafe_allow_html=True)
        for item in ICON_ACTIONS:
            if st.button(item["icon"], key=f"icon_{item['action']}", help=item["label"]):
                _handle_icon_action(item["action"])
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='conversation-header'>최근 대화</div>",
        unsafe_allow_html=True,
    )
    timeline = [
        entry for entry in st.session_state.timeline if entry.get("role") == "user" and entry.get("content")
    ]

    if not timeline:
        st.markdown(
            "<div class='conversation-empty'>대화가 비어 있습니다. 오른쪽 입력창에서 질문을 시작해보세요.</div>",
            unsafe_allow_html=True,
        )
        return

    for entry in reversed(timeline[-8:]):
        try:
            ts = datetime.fromisoformat(entry["ts"])
        except Exception:
            ts = datetime.now(timezone.utc)
        snippet = html.escape(entry["content"][:110] + ("..." if len(entry["content"]) > 110 else ""))
        time_label = ts.astimezone().strftime("%m/%d %H:%M")
        st.markdown(
            f"""
            <div class="conversation-item">
                <div>{snippet}</div>
                <div style="margin-top:0.4rem;font-size:0.75rem;color:#6b7280;">{time_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_header():
    stats = st.session_state.feedback_stats
    st.markdown(
        f"""
        <div class="chat-hero">
            <div class="chat-hero__wrap">
                <div class="chat-hero__icon">{CHIP_ICON}</div>
                <div class="chat-hero__body">
                    <h2>SVC CS Copilot</h2>
                    <p>스크러버 칠러 장비 CS 업무에 특화된 파인튜닝 모델과 Pinecone 문서를 활용해 SOP, 알람, 정산 이슈를 빠르게 해결하세요.</p>
                </div>
                <div class="chat-hero__meta">
                    <span>세션 ID · {st.session_state.session_id[:8]}</span>
                    <span>피드백 · 👍 {stats['like']} | 👎 {stats['bad']}</span>
                </div>
            </div>
        </div>
        <div class="suggestion-row">
          <span class="chip">SOP 점검</span>
          <span class="chip">알람 코드 분석</span>
          <span class="chip">자재 & KIT</span>
          <span class="chip">정산/보고</span>
          <span class="chip">장비 유지보수</span>
          <span class="chip">안전/클린룸</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feedback(container: st.container) -> None:
    stats = st.session_state.feedback_stats
    response_id = st.session_state.last_response_id
    last_answer = st.session_state.last_answer_text

    container.empty()
    with container.container():
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.markdown("#### 이번 답변은 만족스러웠나요?", unsafe_allow_html=True)
        cols = st.columns([1, 1, 1.4])
        like_pressed = bad_pressed = False
        used = st.session_state.feedback_used

        if response_id is None:
            with cols[2]:
                st.info("응답이 생성되면 만족도를 남길 수 있습니다.")
            st.metric("누적 피드백", f"👍 {stats['like']} / 👎 {stats['bad']}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        like_key = f"like_{response_id}"
        bad_key = f"bad_{response_id}"
        already = response_id in used

        with cols[0]:
            like_pressed = st.button("👍 좋아요", key=like_key, disabled=already)
        with cols[1]:
            bad_pressed = st.button("👎 아쉬워요", key=bad_key, disabled=already)
        with cols[2]:
            summary = last_answer[:90] + ("..." if last_answer and len(last_answer) > 90 else "")
            st.metric("누적 피드백", f"👍 {stats['like']} / 👎 {stats['bad']}")
            if summary:
                st.caption(f"최근 답변: {summary}")

        if like_pressed and not already:
            stats["like"] += 1
            used.add(response_id)
            st.success("피드백 감사합니다! 더 도움이 되도록 지속 개선하겠습니다.")
        elif bad_pressed and not already:
            stats["bad"] += 1
            used.add(response_id)
            st.warning("피드백 감사합니다. 더 정확한 답변을 위해 개선하겠습니다.")
        elif already:
            st.info("이미 피드백을 남겨주셨습니다. 감사드립니다!")

        st.markdown("</div>", unsafe_allow_html=True)


def _render_icon_alert(container: st.container) -> None:
    alert = st.session_state.get("icon_alert")
    if not alert:
        return
    level, message = alert
    with container:
        if level == "success":
            st.success(message, icon="✅")
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)
    st.session_state.icon_alert = None


def _render_icon_modal(container: st.container) -> None:
    modal = st.session_state.get("icon_modal")
    if not modal:
        return

    with container:
        st.markdown('<div class="icon-panel">', unsafe_allow_html=True)
        if modal == "recommend":
            st.markdown("#### 추천 프롬프트")
            for prompt in PROMPT_SUGGESTIONS:
                st.markdown(f"- {prompt}")
        elif modal == "favorites":
            st.markdown("#### 즐겨찾기")
            st.info("즐겨찾기 기능을 준비 중입니다. 곧 특정 질의를 고정해둘 수 있어요.")
        elif modal == "schedule":
            st.markdown("#### 일정 & 점검 캘린더")
            st.markdown(
                "- 주간 CS 인수인계 미팅: 매주 수요일 10:00\n"
                "- 월간 품질 점검: 매월 첫째주 화요일 14:00\n"
                "- Pinecone 인덱스 동기화: 매주 월요일 18:00"
            )
        elif modal == "files":
            st.markdown("#### 문서 & 데이터")
            st.code(
                "data/raw/\n"
                "data/processed/\n"
                "scripts/ingest_pinecone.py",
                language="text",
            )
            st.markdown(
                "[🌐 Pinecone Console 바로가기](https://app.pinecone.io)",
                unsafe_allow_html=True,
            )
        elif modal == "settings":
            st.markdown("#### 환경 설정")
            st.json(
                {
                    "모델": os.getenv("MODEL_FT") or os.getenv("SVC_MODEL"),
                    "Pinecone Index": os.getenv("PINECONE_INDEX"),
                    "Temperature": st.session_state.get("icon_temp_preview", "sidebar 참고"),
                }
            )
        st.button("닫기", key=f"close_{modal}", on_click=_close_icon_modal, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _init_states()
    sidebar_cfg = _render_sidebar()
    os.environ["SVC_MODEL"] = sidebar_cfg["model"]
    os.environ["SVC_TEMPERATURE"] = str(sidebar_cfg["temperature"])
    st.session_state["icon_temp_preview"] = sidebar_cfg["temperature"]

    left_col, right_col = st.columns([0.45, 3.2], gap="large")

    history_container = left_col.container()
    with history_container:
        _render_history_panel()
    alert_container = left_col.container()
    modal_container = left_col.container()
    _render_icon_alert(alert_container)
    _render_icon_modal(modal_container)

    header_container = right_col.container()
    with header_container:
        _render_header()

    chat_container = right_col.container()
    with chat_container:
        for role, content in st.session_state.messages:
            with st.chat_message(role):
                st.markdown(content)

    feedback_container = right_col.container()
    _render_feedback(feedback_container)

    prompt = "SOP·알람·불량·정산 등 SVC 이슈를 입력하세요..."
    user_msg = st.chat_input(prompt)

    if not user_msg:
        return

    _append_message("user", user_msg)
    with chat_container.chat_message("user"):
        st.markdown(user_msg)

    history_payload = _history_payload(st.session_state.messages[:-1])
    final_answer = ""
    docs: List[Document] = []

    if sidebar_cfg["use_rag"]:
        if not sidebar_cfg["pinecone_index"]:
            with chat_container.chat_message("assistant"):
                st.error("Pinecone 인덱스가 설정되지 않았습니다. 사이드바에서 Index를 입력하세요.")
            final_answer = ""
        else:
            try:
                chain, retriever = _get_rag_components(
                    sidebar_cfg["pinecone_index"],
                    sidebar_cfg["top_k"],
                    sidebar_cfg["use_multi_query"],
                    sidebar_cfg["use_compression"],
                )
                payload = {"question": user_msg, "history": history_payload}
                with chat_container.chat_message("assistant"):
                    placeholder = st.empty()
                    sources_box = st.container()
                    buf: List[str] = []
                    try:
                        for chunk in chain.stream(payload):  # type: ignore[attr-defined]
                            text = _extract_text(chunk)
                            if not text:
                                continue
                            buf.append(text)
                            placeholder.markdown("".join(buf))
                    except Exception as stream_err:
                        placeholder.warning(f"스트리밍 중단: {stream_err}")

                    if buf:
                        final_answer = "".join(buf)
                    else:
                        result = chain.invoke(payload)  # type: ignore[attr-defined]
                        final_answer = result if isinstance(result, str) else _extract_text(result)
                        placeholder.markdown(final_answer)

                    if sidebar_cfg["show_sources"]:
                        try:
                            docs = retriever.get_relevant_documents(user_msg)  # type: ignore[assignment]
                            if docs:
                                sources_box.markdown(_render_sources(docs), unsafe_allow_html=True)
                        except Exception as retr_err:
                            sources_box.warning(f"출처 조회 실패: {retr_err}")
            except Exception as err:
                with chat_container.chat_message("assistant"):
                    st.error(f"RAG 체인 동작 중 오류가 발생했습니다: {err}")
                final_answer = ""
    else:
        chat = get_chat(model=sidebar_cfg["model"], temperature=sidebar_cfg["temperature"], streaming=True)
        with chat_container.chat_message("assistant"):
            placeholder = st.empty()
            buf: List[str] = []
            try:
                for chunk in chat.stream(user_msg):
                    text = _extract_text(chunk)
                    if not text:
                        continue
                    buf.append(text)
                    placeholder.markdown("".join(buf))
            except Exception as stream_err:
                placeholder.warning(f"스트리밍 중단: {stream_err}")
            answer = chat.invoke(user_msg)
            final_answer = "".join(buf) if buf else _extract_text(answer)
            placeholder.markdown(final_answer)
            docs = []

    final_answer = final_answer.strip() or "(응답 없음)"
    _append_message("assistant", final_answer)
    st.session_state.last_response_id = len(st.session_state.messages)
    st.session_state.last_answer_text = final_answer

    history_container.empty()
    with history_container:
        _render_history_panel()
    alert_container.empty()
    modal_container.empty()
    _render_icon_alert(alert_container)
    _render_icon_modal(modal_container)

    _render_feedback(feedback_container)


if __name__ == "__main__":
    main()
