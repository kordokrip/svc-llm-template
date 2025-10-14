
import os
import json
import uuid
import time
import requests
import streamlit as st
from typing import Dict, Any, Iterable, Optional

# -----------------------------------------------------------------------------
# Page & Session
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SVC AI Chat", layout="wide")
st.title("💬 SVC AI Chat")

if "history" not in st.session_state:
    st.session_state.history = []  # [{role, content}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "trace_id" not in st.session_state:
    st.session_state.trace_id = str(uuid.uuid4())

# History cap (ENV overrideable)
MAX_HISTORY = int(os.getenv("SVC_CHAT_MAX_HISTORY", "16"))

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 설정")
    api_base = os.getenv("API_URL", "http://127.0.0.1:8000")
    api_base = st.text_input("API Base URL", api_base)

    col_a, col_b = st.columns(2)
    with col_a:
        path_chat = st.text_input("/chat 경로", value="/chat")
    with col_b:
        path_stream = st.text_input("/chat 스트림 경로", value="/chat/stream")

    streaming = st.checkbox("스트리밍 사용", value=True, help="서버가 SSE/Chunk를 지원하면 토큰 단위로 표시합니다.")
    timeout = st.slider("요청 타임아웃(초)", min_value=10, max_value=300, value=120, step=5)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.history = []
            st.success("초기화 완료")
    with c2:
        if st.button("대화 JSON 내보내기", use_container_width=True):
            st.session_state["_export_click"] = True

# Export (separate from reruns)
if st.session_state.get("_export_click"):
    st.session_state["_export_click"] = False
    export_name = f"svc_chat_{int(time.time())}.json"
    st.download_button(
        "💾 대화 JSON 다운로드",
        data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
        file_name=export_name,
        mime="application/json",
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _headers() -> Dict[str, str]:
    return {
        "X-Trace-Id": st.session_state.trace_id,
        "X-Session-Id": st.session_state.session_id,
        "Content-Type": "application/json",
    }


def _safe_join(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def _trim_history():
    h = st.session_state.history
    if len(h) > MAX_HISTORY:
        st.session_state.history = h[-MAX_HISTORY:]


def _post_chat(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    r = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"answer": r.text}


def _iter_stream(url: str, payload: Dict[str, Any], timeout: int) -> Iterable[str]:
    """Yield text chunks from SSE/JSONL/text stream.

    Supported formats:
    - **SSE**: lines starting with 'data: '
    - **JSONL**: one JSON object per line with keys {delta|answer}
    - **Plain text**: raw chunks via iter_content
    """
    with requests.post(url, headers=_headers(), json=payload, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        # Try line-based first (SSE or JSONL)
        for raw in resp.iter_lines(decode_unicode=True):  # keeps connection open
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            # SSE (data: ...)
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if not data:
                    continue
                # Try JSON payload {delta|answer}
                try:
                    obj = json.loads(data)
                    if isinstance(obj, dict):
                        if "delta" in obj and obj["delta"]:
                            yield str(obj["delta"])
                        elif "answer" in obj and obj["answer"]:
                            yield str(obj["answer"])
                        continue
                except Exception:
                    # It's plain text after data:
                    yield data
                    continue
            # JSONL
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if "delta" in obj and obj["delta"]:
                        yield str(obj["delta"])
                    elif "answer" in obj and obj["answer"]:
                        yield str(obj["answer"])
                    continue
            except Exception:
                pass
            # Fallback: emit line as-is
            yield line

        # If server didn't send line-based chunks, fallback to raw bytes
        if not ctype or "text/plain" in ctype:
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk


# -----------------------------------------------------------------------------
# Chat loop
# -----------------------------------------------------------------------------
user = st.chat_input("무엇을 도와드릴까요? (SOP/자재/KIT/정산/보고)")
if user:
    st.session_state.history.append({"role": "user", "content": user})
    _trim_history()

    with st.chat_message("user"):
        st.markdown(user)

    # Assistant container for streaming / non-streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text: Optional[str] = None

        payload = {
            "question": user,
            "history": st.session_state.history,
            "session_id": st.session_state.session_id,
        }

        if streaming:
            status = st.status("서버 연결 중...", expanded=False)
            try:
                url = _safe_join(api_base, path_stream)
                buf = []
                for chunk in _iter_stream(url, payload, timeout):
                    buf.append(chunk)
                    # For markdown stability, join softly
                    text = "".join(buf)
                    placeholder.markdown(text)
                final_text = "".join(buf).strip() or "(빈 응답)"
                status.update(label="완료", state="complete")
            except requests.HTTPError as e:
                status.update(label="스트리밍 실패 — HTTP 오류", state="error")
                final_text = f"요청 실패(HTTP): {e}"
                placeholder.error(final_text)
            except requests.RequestException as e:
                status.update(label="스트리밍 실패 — 네트워크 오류", state="error")
                final_text = f"요청 실패(네트워크): {e}"
                placeholder.error(final_text)
            except Exception as e:
                status.update(label="스트리밍 실패 — 예외", state="error")
                final_text = f"요청 실패(예외): {e}"
                placeholder.error(final_text)
        else:
            try:
                url = _safe_join(api_base, path_chat)
                data = _post_chat(url, payload, timeout)
                final_text = str(data.get("answer") or "(응답 없음)")
                placeholder.markdown(final_text)

                # Optional: show sources if server returns them
                sources = data.get("sources")
                if isinstance(sources, list) and sources:
                    with st.expander("🔗 참고 자료"):
                        for s in sources:
                            title = s.get("title") or s.get("id") or "출처"
                            href = s.get("url") or s.get("source")
                            if href:
                                st.markdown(f"- [{title}]({href})")
                            else:
                                st.markdown(f"- {title}")

            except requests.HTTPError as e:
                final_text = f"요청 실패(HTTP): {e}"
                placeholder.error(final_text)
            except requests.RequestException as e:
                final_text = f"요청 실패(네트워크): {e}"
                placeholder.error(final_text)
            except Exception as e:
                final_text = f"요청 실패(예외): {e}"
                placeholder.error(final_text)

    # Update history with assistant turn
    final_text = final_text or "(응답 없음)"
    st.session_state.history.append({"role": "assistant", "content": final_text})
    _trim_history()
