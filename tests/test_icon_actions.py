import sys
import types
from pathlib import Path

import pytest

# Stub external dependencies so ChatApp imports cleanly in test context
if "langchain_core" not in sys.modules:
    lc = types.ModuleType("langchain_core")
    documents = types.ModuleType("documents")
    documents.Document = object
    messages = types.ModuleType("messages")
    messages.AIMessage = messages.AIMessageChunk = messages.BaseMessage = object
    lc.documents = documents
    lc.messages = messages
    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.documents"] = documents
    sys.modules["langchain_core.messages"] = messages

if "svc_llm" not in sys.modules:
    svc_llm = types.ModuleType("svc_llm")
    sys.modules["svc_llm"] = svc_llm

    llm = types.ModuleType("svc_llm.llm")
    sys.modules["svc_llm.llm"] = llm
    chat_model = types.ModuleType("svc_llm.llm.chat_model")

    def _dummy_get_chat(*args, **kwargs):
        class _Dummy:
            def stream(self, *_args, **_kwargs):
                yield "dummy"

            def invoke(self, *_args, **_kwargs):
                return "dummy"

        return _Dummy()

    chat_model.get_chat = _dummy_get_chat
    sys.modules["svc_llm.llm.chat_model"] = chat_model

    rag = types.ModuleType("svc_llm.rag")
    sys.modules["svc_llm.rag"] = rag
    rag_chain = types.ModuleType("svc_llm.rag.rag_chain_pinecone")

    def _dummy_build_chain_pinecone(*args, **kwargs):
        class _Chain:
            def stream(self, *_args, **_kwargs):
                yield "dummy"

            def invoke(self, *_args, **_kwargs):
                return "dummy"

        return _Chain()

    rag_chain.build_chain_pinecone = _dummy_build_chain_pinecone
    sys.modules["svc_llm.rag.rag_chain_pinecone"] = rag_chain

    vectorstore = types.ModuleType("svc_llm.vectorstore")
    sys.modules["svc_llm.vectorstore"] = vectorstore
    pinecone_store = types.ModuleType("svc_llm.vectorstore.pinecone_store")
    pinecone_store.get_pinecone = lambda *args, **kwargs: None
    sys.modules["svc_llm.vectorstore.pinecone_store"] = pinecone_store

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import ChatApp as app  # noqa: E402


class DummySession(dict):
    def __getattr__(self, item):  # pragma: no cover - mirrors Streamlit behaviour
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture(autouse=True)
def session_state(monkeypatch):
    state = DummySession()
    dummy_st = types.SimpleNamespace(session_state=state)
    monkeypatch.setattr(app, "st", dummy_st, raising=False)
    app._init_states()
    yield state


def test_new_chat_resets_state(session_state):
    session_state.messages.append(("user", "test"))
    session_state.timeline.append({"role": "user", "content": "test", "ts": ""})
    app._handle_icon_action("new_chat")
    assert session_state.messages == []
    assert session_state.icon_modal is None
    assert session_state.icon_alert[0] == "success"


def test_recommend_sets_modal(session_state):
    app._handle_icon_action("recommend")
    assert session_state.icon_modal == "recommend"


def test_favorites_sets_modal_and_alert(session_state):
    app._handle_icon_action("favorites")
    assert session_state.icon_modal == "favorites"
    alert = session_state.icon_alert
    assert alert and "즐겨찾기" in alert[1]


def test_schedule_sets_modal(session_state):
    app._handle_icon_action("schedule")
    assert session_state.icon_modal == "schedule"


def test_schedule_calendar_html_contains_event():
    html = app._render_schedule_calendar_html()
    assert "SCV 데이터 업데이트 예정" in html
    assert "highlight" in html


def test_files_sets_modal(session_state):
    app._handle_icon_action("files")
    assert session_state.icon_modal == "files"


def test_settings_sets_modal(session_state):
    app._handle_icon_action("settings")
    assert session_state.icon_modal == "settings"
