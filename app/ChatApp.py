# --- ensure 'src' on sys.path for Streamlit Cloud ---
import os, sys
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------
import calendar
import html
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from svc_llm.llm.chat_model import get_chat
from svc_llm.rag.rag_chain_pinecone import build_chain_pinecone
# NOTE: Avoid importing `get_pinecone` at module import time to prevent hard crashes on Streamlit Cloud
# when the Pinecone SDK is misinstalled (e.g., `pinecone-client` instead of `pinecone`).

load_dotenv()

st.set_page_config(page_title="GST CS AI ChatBot", page_icon="💬", layout="wide")
st.title("")
st.markdown(
    """
    <style>
    /* 전체 앱 폰트 및 배경 그라디언트를 지정하여 부드러운 색감을 유지 */
    body, .stApp {
        font-family: 'Plus Jakarta Sans', 'Pretendard', sans-serif;
        background: radial-gradient(circle at 14% 20%, rgba(196,226,255,0.45), transparent 52%),
                    radial-gradient(circle at 80% 10%, rgba(242,210,255,0.45), transparent 55%),
                    linear-gradient(135deg, #f9fbff 0%, #f5f0ff 45%, #fef8f4 100%);
        color: #102036;
    }
    :root{
        --container-max: clamp(960px, 86vw, 1480px);
        --pad-x: clamp(12px, 3vw, 40px);
        --pad-b: clamp(24px, 4vw, 48px);
    }
    div.block-container {
        padding: 0 var(--pad-x) var(--pad-b) var(--pad-x);
        max-width: var(--container-max);
        margin: 0 auto;
    }
    /* 사이드바 영역 배경을 반투명 처리하고 경계선 추가 */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(18px);
        color: #1e293b;
        padding-top: 1.5rem;
        border-right: 1px solid rgba(15,23,42,0.08);
    }
    /* 사이드바 내부 텍스트 기본 컬러와 굵기 */
    section[data-testid="stSidebar"] * {
        color: #2b3445 !important;
        font-weight: 500;
    }
    /* 슬라이더 색상 지정 */
    section[data-testid="stSidebar"] .stSlider [role="slider"] {
        background: linear-gradient(135deg, #38bdf8, #a855f7);
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #475569 !important;
    }
    /* 셀렉트박스 및 텍스트 입력 필드 스타일統一 */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
    section[data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.7);
        border-radius: 14px;
        border: 1px solid rgba(100,116,139,0.18);
    }
    /* 토글 스위치 배경/테두리 */
    section[data-testid="stSidebar"] .stToggle {
        background: rgba(241,245,249,0.7);
        border-radius: 20px;
        padding: 0.45rem 0.6rem;
        border: 1px solid rgba(148,163,184,0.25);
    }
    /* 사이드바 버튼 컬러 및 그림자 */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #a855f7);
        border: none;
        color: #fff !important;
        border-radius: 14px;
        font-weight: 600;
        box-shadow: 0 34px 68px rgba(168,85,247,0.25);
    }
    /* 사이드바 로고(아이콘+텍스트) 정렬 */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0 0.6rem 1.6rem 0.6rem;
    }
    /* 로고 아이콘 배경 및 그림자 */
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
    /* 로고 내부 SVG 크기 */
    .sidebar-chip-icon svg {
        width: 36px;
        height: 36px;
        fill: #f8fafc;
    }
    /* 사이드바 타이틀 텍스트 */
    .sidebar-title {
        margin: 0;
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    /* 사이드바 서브 타이틀 */
    .sidebar-subtitle {
        margin: 0;
        font-size: 0.78rem;
        opacity: 0.65;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    /* 아이콘 스택 기본 구조 */
    .icon-stack {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin: 1rem 0 1.4rem 0;
        padding-left: 0.7rem;  /* 사이드바 아이콘 여백 30% 축소 (요청사항) */
    }
    /* 아이콘 버튼 스타일 */
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
    /* 아이콘 버튼 호버 효과 */
    .icon-toolbar .stButton > button:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.95);
        box-shadow: 0 20px 30px rgba(99,102,241,0.22);
    }
    /* 아이콘 버튼 간 마진 조정 */
    .icon-toolbar .stButton {
        margin-bottom: 0.5rem;
    }
    /* 아이콘 패널(컨텐츠 카드) 스타일 */
    .icon-panel {
        margin-top: 1.1rem;
        padding: 1.3rem 1.55rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 22px 44px rgba(148,163,184,0.2);
        backdrop-filter: blur(14px);
        width: calc(100% + 1.1rem);  /* 사이드 카드 가로폭 30% 축소 */
        margin-left: -0.55rem;
    }
    /* 아이콘 패널 제목 텍스트 */
    .icon-panel h4 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        font-size: 1rem;
        color: #0f172a;
    }
    /* 아이콘 패널 리스트 */
    .icon-panel ul {
        margin: 0;
        padding-left: 1.3rem;
        font-size: 0.94rem;
        line-height: 1.7;
        color: #1f2937;
    }
    /* 일정 캘린더 스타일 */
    .cal-wrap {
        width: 100%;
        background: rgba(236,245,255,0.85);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(148,163,184,0.24);
        box-shadow: inset 0 1px 6px rgba(255,255,255,0.4);
    }
    .cal-header {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.6rem;
        color: #1e3a8a;
    }
    .cal-table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(255,255,255,0.82);
        border-radius: 12px;
        overflow: hidden;
    }
    .cal-table th,
    .cal-table td {
        padding: 0.45rem;
        text-align: center;
        border: 1px solid rgba(148,163,184,0.24);
        font-size: 0.85rem;
    }
    .cal-day.highlight {
        background: rgba(245,158,11,0.18);
        font-weight: 700;
        color: #b45309;
    }
    .cal-empty {
        background: rgba(255,255,255,0.4);
    }
    .cal-note {
        margin-top: 0.6rem;
        font-size: 0.85rem;
        color: #334155;
    }
    /* 개별 사이드 아이콘 스타일 */
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
    /* 최근 대화 제목 라벨 */
    .conversation-header {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }
    /* 최근 대화 카드 요소 */
    .conversation-item {
        padding: 0.85rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.14);
        box-shadow: 0 28px 52px rgba(148,163,184,0.18);
        font-size: 0.9rem;
        line-height: 1.45;
        color: #182230;
        backdrop-filter: blur(12px);
        width: clamp(280px, 70%, 520px); /* ← 기존 대비 약 30% 좁게 */
        margin-left: 0;
        margin-right: auto; /* 왼쪽 정렬 유지 */
    }
    /* 최근 대화 카드 간격 */
    .conversation-item + .conversation-item {
        margin-top: 0.65rem;
    }
    /* 최근 대화가 없을 때 안내 박스 */
    .conversation-empty {
        padding: 1.2rem 1.25rem;
        border-radius: 18px;
        background: rgba(248,250,252,0.85);
        border: 1px dashed rgba(148,163,184,0.4);
        color: #64748b;
        font-size: 0.9rem;
        width: clamp(280px, 70%, 520px); /* ← 최근 대화 카드와 동일 폭 */
        margin-left: 0;
        margin-right: auto;
    }
    /* 메인 히어로 카드 컨테이너 */
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
    /* 히어로 카드 내부 하이라이트 효과 */
    .chat-hero::after {
        content: "";
        position: absolute;
        inset: -35% 50% auto -20%;
        height: 260px;
        background: radial-gradient(circle, rgba(129,212,250,0.45) 0%, transparent 70%);
        pointer-events: none;
    }
    /* 히어로 카드 내부 레이아웃 구조 */
    .chat-hero__wrap {
        position: relative;
        display: flex;
        gap: 1.7rem;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        z-index: 1;
    }
    /* 히어로 영역의 큰 칩 아이콘 꾸미기 */
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
    /* 히어로 아이콘 내부 SVG 크기 조정 */
    .chat-hero__icon svg {
        width: 46px;
        height: 46px;
        fill: #f8fafc;
    }
    /* 히어로 타이틀 텍스트 */
    .chat-hero__body h2 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: #0f172a;
    }
    /* 히어로 본문 영역 넓이 */
    .chat-hero__body {
        flex: 1 1 250px;
    }
    /* 히어로 설명 문장 */
    .chat-hero__body p {
        margin: 0.52rem 0 0 0;
        font-size: 0.98rem;
        color: #334155;
        line-height: 1.6;
    }
    /* 히어로 우측 세션 정보 */
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
    /* 주제 칩 기본 스타일 */
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
    /* 칩 사이 간격 */
    .chip + .chip {
        margin-left: 0.45rem;
    }
    /* 채팅 메시지 버블 스타일 */
    [data-testid="stChatMessage"] {
        border-radius: 22px;
        padding: 1.1rem 1.3rem;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 20px 36px rgba(148,163,184,0.22);
        margin-bottom: 1rem;
        backdrop-filter: blur(14px);
    }
    /* 어시스턴트 버블 색상 */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: rgba(240,249,255,0.92);
    }
    /* 메시지 텍스트 가독성 */
    [data-testid="stChatMessage"] pre, [data-testid="stChatMessage"] p {
        font-size: 1.02rem;
        line-height: 1.65;
        color: #0f172a;
    }
    /* 근거 출처 뱃지 */
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
    /* 피드백 카드 컨테이너 */
    .feedback-box {
        margin-top: 1.6rem;
        padding: 1.15rem 1.3rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 18px 36px rgba(148,163,184,0.18);
    }
    /* 피드백 카드 제목 */
    .feedback-box h4 {
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #1f2937;
    }
    /* 피드백 버튼 */
    .feedback-box button {
        width: 100%;
        border-radius: 18px !important;
        padding: 0.45rem 0 !important;
        font-weight: 600 !important;
        border: none !important;
    }
    /* 주제 칩 행 레이아웃 */
    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 1.1rem;
        align-items: center;
    }

    /* ===== Responsive container & chat widths ===== */
    @supports (width: clamp(1px, 1vw, 1000px)) {
        [data-testid="stChatMessage"],
        [data-testid="stChatInput"] > div,
        .chat-hero {
            max-width: clamp(560px, 60vw, 1100px);
            margin-left: auto;
            margin-right: auto;
        }
    }

    /* Wider screens: relax container max slightly */
    @media (min-width: 1400px){
        :root{ --container-max: clamp(1120px, 88vw, 1600px); }
    }
    @media (min-width: 1700px){
        :root{ --container-max: clamp(1280px, 90vw, 1760px); }
    }

    /* Narrow screens: compact paddings and allow full width chat */
    @media (max-width: 768px){
        :root{ --pad-x: clamp(8px, 4vw, 20px); }
        [data-testid="stChatMessage"],
        [data-testid="stChatInput"] > div {
            max-width: 100%;
        }
    }
    /* 입력창 외곽 스타일 (더 둥글고 강조) */
    [data-testid="stChatInput"] > div {
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(235,245,255,0.92));
        border-radius: 42px;
        border: 1.5px solid rgba(79,70,229,0.3);
        box-shadow: 0 32px 56px rgba(79,70,229,0.24);
        padding: 0.85rem 1.1rem;
    }
    /* 입력창 텍스트 스타일 */
    [data-testid="stChatInput"] textarea {
        font-size: 1.12rem;
        color: #0b1f40;
        font-weight: 600;
        padding: 0.25rem 0.4rem;
        background: transparent;
    }
    /* 입력창 플레이스홀더 스타일 */
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(15,23,42,0.38);
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    /* ====== [1.2 요청사항] 응답 카드/섹션 헤더 스타일 ====== */
    /* 채팅 메시지 내부 섹션 제목(요약/해결 단계/주의·제약)에 가독성 강화 */
    [data-testid="stChatMessage"] h3 {
        margin: 0.2rem 0 0.6rem 0;
        padding: 0.35rem 0.6rem;
        border-left: 6px solid #f59e0b; /* 오렌지 강조 */
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,247,237,0.9));
        border-radius: 10px;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    [data-testid="stChatMessage"] h4 {
        margin: 0.35rem 0 0.4rem 0;
        padding: 0.25rem 0.5rem;
        border-left: 4px solid #60a5fa; /* 블루 보조 강조 */
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(236,252,255,0.9));
        border-radius: 8px;
        font-weight: 700;
    }
    /* 본문 리스트 가독성 향상 */
    [data-testid="stChatMessage"] ol {
        padding-left: 1.2rem;
        margin: 0.2rem 0 0.6rem 0;
        line-height: 1.75;
    }
    [data-testid="stChatMessage"] ul {
        padding-left: 1.1rem;
        margin: 0.2rem 0 0.6rem 0;
        line-height: 1.7;
    }
    /* 강조 배지(요약 박스 느낌) */
    [data-testid="stChatMessage"] .badge {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(245,158,11,0.35);
        background: rgba(245,158,11,0.12);
        color: #92400e;
        font-weight: 700;
        font-size: 0.8rem;
    }
    /* 경고/주의 박스 느낌 */
    [data-testid="stChatMessage"] .warn-box {
        margin: 0.8rem 0 0.6rem 0;
        padding: 0.8rem 0.95rem;
        border-radius: 12px;
        background: rgba(254,243,199,0.6);
        border: 1px solid rgba(245,158,11,0.3);
        color: #7c2d12;
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

ICON_STACK = [
    ("＋", "새 대화"),
    ("🧭", "추천"),
    ("⭐", "즐겨찾기"),
    ("🗓️", "일정"),
    ("📎", "파일"),
    ("⚙️", "설정"),
]

# 아이콘 라벨별 액션 매핑(요청: 아이콘 클릭 시 기능 수행)
ICON_ACTION_MAP = {
    "새 대화": "new_chat",
    "추천": "recommend",
    "즐겨찾기": "favorites",
    "일정": "schedule",
    "파일": "files",
    "설정": "settings",
}
PROMPT_SUGGESTIONS = [
    "SOP | 챔버 온도 이탈 시 1차 대응 프로세스는?",
    "알람 | Heater TS EG0-60-07 발생 원인과 조치 순서?",
    "자재 | EBR KIT 교체 주기와 재고 확보 체크리스트?",
    "정산 | 최근 3개월 미해결 CS건 요약 및 비용 추산?",
    "보고 | 주간 품질 리포트에 포함해야 할 핵심 KPI는?",
]


def _render_schedule_calendar_html() -> str:
    today = datetime.now()
    year, month = today.year, today.month
    cal = calendar.monthcalendar(year, month)
    header = calendar.month_name[month]
    rows = []
    for week in cal:
        cells = []
        for day in week:
            if day == 0:
                cells.append("<td class='cal-empty'></td>")
            elif day == 15:
                cells.append(f"<td class='cal-day highlight'>{day}</td>")
            else:
                cells.append(f"<td class='cal-day'>{day}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    table = (
        f"<div class='cal-wrap'><div class='cal-header'>{year}년 {header}</div>"
        "<table class='cal-table'><thead><tr>"
        + "".join(f"<th>{d}</th>" for d in ["월","화","수","목","금","토","일"])
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "<div class='cal-note'>매월 15일 · SCV 데이터 업데이트 예정</div></div>"
    )
    return table


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
    if "icon_toolbar_iter" not in st.session_state:
        st.session_state.icon_toolbar_iter = 0  # 아이콘 버튼 key 중복 방지용


def _reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.timeline = []
    st.session_state.last_response_id = None
    st.session_state.last_answer_text = ""
    st.session_state.icon_modal = None
    st.session_state.icon_alert = None
    st.session_state.pop("queued_question", None)


def _handle_icon_action(action: str) -> None:
    if action == "new_chat":
        _reset_conversation()
        st.session_state.icon_alert = ("success", "새 대화가 시작되었습니다. 오른쪽 입력창에서 질문을 시작하세요.")
        return
    if action == "favorites":
        st.session_state.icon_modal = action
        st.session_state.icon_alert = (
            "info",
            "브라우저 즐겨찾기는 Ctrl+D (macOS는 ⌘+D) 단축키로 추가할 수 있습니다.",
        )
        return
    if action in {"recommend", "schedule", "files", "settings"}:
        st.session_state.icon_modal = action
        return
    st.session_state.icon_modal = None


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

    # Lazy import to catch dependency issues gracefully on Streamlit Cloud
    try:
        from svc_llm.vectorstore.pinecone_store import get_pinecone
    except Exception as e:
        st.error(
            "Pinecone SDK 임포트 오류가 발생했습니다. Cloud 빌드에서 `pinecone-client`(구버전)과 `pinecone`(신버전) 혼재 시 자주 발생합니다.\n"
            "해결: requirements.txt 에서 `pinecone-client` 를 제거하고 `pinecone>=3` 와(필요 시) `langchain-pinecone` 를 추가하세요.",
            icon="🧩",
        )
        st.stop()

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

        if st.button("🔧 단위 테스트 실행", use_container_width=True):
            results = _run_unit_tests()
            ok_cnt = sum(1 for r in results if r.startswith("✅"))
            fail_cnt = sum(1 for r in results if r.startswith("❌"))
            st.info(f"테스트 결과 (성공/실패): {ok_cnt} / {fail_cnt}")
            for r in results:
                st.write(r)

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
        suffix = st.session_state.get("icon_toolbar_iter", 0)  # 아이콘 버튼 키 고유화(요청사항)
        for icon, label in ICON_STACK:
            action = ICON_ACTION_MAP.get(label)
            if st.button(icon, key=f"icon_{action}_{suffix}", help=label):
                if action:
                    _handle_icon_action(action)
        st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.icon_toolbar_iter += 1  # 다음 렌더링 시 중복 키 방지
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
                    <h2>SVC CS 인공지능 언어모델 Chart GPT </h2>
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
            for idx, prompt in enumerate(PROMPT_SUGGESTIONS, 1):
                if st.button(prompt, key=f"suggest_{idx}"):
                    st.session_state["queued_question"] = prompt
                    st.session_state.icon_modal = None
                    st.session_state.icon_alert = ("info", f"추천 질문을 전송합니다: {prompt}")
                    st.experimental_rerun()
        elif modal == "favorites":
            st.markdown("#### 즐겨찾기 안내")
            components.html(
                """
                <button id='bookmark-btn' style='padding:0.6rem 1rem;border:none;border-radius:12px;background:#4f46e5;color:white;font-weight:600;cursor:pointer;'>북마크 추가 시도</button>
                <script>
                const btn = document.getElementById('bookmark-btn');
                btn.addEventListener('click', function(){
                    const url = window.location.href;
                    const title = document.title || 'GST CS Copilot';
                    try {
                        if (window.sidebar && window.sidebar.addPanel) {
                            window.sidebar.addPanel(title, url, '');
                        } else if (window.external && ('AddFavorite' in window.external)) {
                            window.external.AddFavorite(url, title);
                        } else {
                            alert('즐겨찾기 단축키: Ctrl + D (macOS는 ⌘ + D)를 눌러주세요.');
                        }
                    } catch (err) {
                        alert('즐겨찾기 단축키: Ctrl + D (macOS는 ⌘ + D)를 눌러주세요.');
                    }
                });
                </script>
                """,
                height=120,
            )
            st.markdown(
                "- 북마크 이름은 `GST CS Copilot`으로 저장하는 것을 권장합니다.  \n"
                "- 모바일 브라우저는 공유 메뉴에서 `홈 화면 추가` 또는 `즐겨찾기 추가`를 선택하세요."
            )
        elif modal == "schedule":
            st.markdown("#### 일정 & 점검 캘린더")
            st.markdown(_render_schedule_calendar_html(), unsafe_allow_html=True)
        elif modal == "files":
            st.markdown("#### 파일 기능")
            st.info("추후 사용자가 채팅을 한 내용의 데이터를 문서형식으로 다운 받을 수 있는 기능 개발 중:")
        elif modal == "settings":
            st.markdown("#### 설정 안내")
            st.info("환경설정에서 관리자가 값읋 조정할 수 있습니다.")
        st.button("닫기", key=f"close_{modal}", on_click=_close_icon_modal, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _run_unit_tests() -> List[str]:
    """Runs lightweight in-app unit tests (no external services)."""
    results: List[str] = []

    def _ok(name: str, cond: bool):
        results.append(("✅ " + name) if cond else ("❌ " + name))

    # 1) ICON_ACTION_MAP completeness
    try:
        expected = {
            "new_chat",
            "recommend",
            "favorites",
            "schedule",
            "files",
            "settings",
        }
        _ok("ICON_ACTION_MAP has all actions", set(ICON_ACTION_MAP.values()) == expected)
    except Exception:
        _ok("ICON_ACTION_MAP has all actions", False)

    # 2) New chat clears history
    try:
        _init_states()
        st.session_state.messages = [("user", "hello"), ("assistant", "hi")]  # seed
        _handle_icon_action("new_chat")
        _ok("new_chat clears conversation", len(st.session_state.messages) == 0)
        _ok("new_chat sets success alert", st.session_state.get("icon_alert", (None, None))[0] == "success")
    except Exception:
        _ok("new_chat clears conversation", False)
    # keep alert test even if above failed
    try:
        _ok("new_chat sets success alert", st.session_state.get("icon_alert", (None, None))[0] == "success")
    except Exception:
        _ok("new_chat sets success alert", False)

    # 3) Favorites opens modal
    try:
        _handle_icon_action("favorites")
        _ok("favorites opens modal", st.session_state.get("icon_modal") == "favorites")
    except Exception:
        _ok("favorites opens modal", False)

    # 4) Calendar HTML contains 15th highlight and SCV note
    try:
        html_str = _render_schedule_calendar_html()
        _ok("calendar has highlight class", "highlight" in html_str)
        _ok("calendar note mentions SCV update", "SCV 데이터 업데이트 예정" in html_str)
    except Exception:
        _ok("calendar has highlight class", False)
        _ok("calendar note mentions SCV update", False)

    # 5) History payload filters to chat roles
    try:
        payload = _history_payload([("user", "u"), ("assistant", "a"), ("system", "s"), ("other", "x")])
        roles = [m["role"] for m in payload]
        _ok("history payload keeps only chat roles", roles == ["user", "assistant", "system"])
    except Exception:
        _ok("history payload keeps only chat roles", False)

    # 6) _extract_text handles AIMessage
    try:
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="hello")
        _ok("_extract_text for AIMessage", _extract_text(msg) == "hello")
    except Exception:
        _ok("_extract_text for AIMessage", False)

    return results


def main() -> None:
    _init_states()
    sidebar_cfg = _render_sidebar()
    os.environ["SVC_MODEL"] = sidebar_cfg["model"]
    os.environ["SVC_TEMPERATURE"] = str(sidebar_cfg["temperature"])
    st.session_state["icon_temp_preview"] = sidebar_cfg["temperature"]

    queued_question = st.session_state.pop("queued_question", None)

    left_col, right_col = st.columns([1.6, 2.4], gap="small")  # 사이드 영역 가로폭 재조정 (요청사항)

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
    if not user_msg and queued_question:
        user_msg = queued_question
    elif queued_question:
        st.session_state["queued_question"] = queued_question

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
                    placeholder.info("💭 생각 중 입니다...", icon="⏳")  # LLM 응답 대기 메시지
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
            placeholder.info("💭 생각 중 입니다...", icon="⏳")  # LLM 응답 대기 메시지
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