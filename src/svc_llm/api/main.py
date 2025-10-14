from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from ..logging import setup_logging
from ..vectorstore.chroma_store import get_chroma
# pinecone backend is optional – import lazily inside a helper to avoid hard dependency
# from ..vectorstore.pinecone_store import get_pinecone  # (imported inside function)
from ..rag.retriever import make_retriever
from ..llm.chain import build_chain
from dotenv import load_dotenv

# --- Environment & logging ----------------------------------------------------
load_dotenv()
log = setup_logging()

SVC_VECTOR_BACKEND = os.getenv("SVC_VECTOR_BACKEND", "chroma").lower()  # chroma|pinecone
SVC_RETRIEVAL_K = int(os.getenv("SVC_RETRIEVAL_K", "4"))
SVC_MODEL = os.getenv("SVC_MODEL", "ft:gpt-4.1-mini-2025-04-14:personal:svc-41mini-sft-dpo-80usd-sft:CLB4qudK")
ALLOW_ORIGINS = [o.strip() for o in os.getenv("SVC_CORS_ORIGINS", "*").split(",") if o.strip()]

# --- Data models --------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatIn(BaseModel):
    question: str = Field(..., description="현재 사용자 질의")
    history: List[Message] = Field(default_factory=list, description="대화 히스토리")
    top_k: Optional[int] = Field(default=None, description="Retriever k override")

class ChatOut(BaseModel):
    answer: str

# --- Helpers ------------------------------------------------------------------

def _get_vectorstore_from_env():
    """Return a VectorStore instance based on env config.
    Defaults to Chroma. Pinecone path is lazy to keep optional dependency.
    """
    backend = SVC_VECTOR_BACKEND
    if backend == "chroma":
        return get_chroma()
    elif backend == "pinecone":
        # Lazy import to avoid hard dependency when user doesn't need Pinecone
        from ..vectorstore.pinecone_store import get_pinecone  # type: ignore
        return get_pinecone()
    else:
        raise RuntimeError(f"Unknown vector backend: {backend}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create heavy resources once and keep them in app.state.
    Using lifespan is the recommended pattern in FastAPI for startup/shutdown.
    """
    try:
        vs = _get_vectorstore_from_env()
        retriever = make_retriever(vs, k=SVC_RETRIEVAL_K)
        chain = build_chain(retriever, model_name=SVC_MODEL)
        app.state.vectorstore = vs
        app.state.retriever = retriever
        app.state.chain = chain
        log.info("App initialized (backend=%s, k=%s, model=%s)", SVC_VECTOR_BACKEND, SVC_RETRIEVAL_K, SVC_MODEL)
        yield
    finally:
        # If the vector store has a close() or similar, call it here
        for attr in ("vectorstore", "retriever", "chain"):
            if hasattr(app.state, attr):
                # No-op; kept for symmetry and future resource cleanup
                pass

app = FastAPI(title="SVC LLM API", lifespan=lifespan)

# CORS – allow all by default, can be restricted via SVC_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependencies -------------------------------------------------------------

def get_chain(request: Request):
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(status_code=503, detail="Chain is not initialized yet.")
    return chain

# --- Core logic ---------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _invoke_chain_sync(chain, question: str, history: List[Message], top_k: Optional[int]):
    payload = {
        "question": question,
        "history": [h.model_dump() for h in history],
    }
    if top_k is not None and hasattr(chain, "with_config"):
        # If your build_chain supports dynamic retriever k, thread it via config
        payload["top_k"] = top_k
    return chain.invoke(payload)

# --- Routes -------------------------------------------------------------------
@app.get("/healthz")
async def healthz(request: Request):
    return {
        "status": "ok",
        "backend": SVC_VECTOR_BACKEND,
        "k": SVC_RETRIEVAL_K,
        "model": SVC_MODEL,
        "has_chain": bool(getattr(request.app.state, "chain", None)),
    }

@app.post("/chat", response_model=ChatOut)
async def chat(in_: ChatIn, chain = Depends(get_chain)):
    log.info("question=%s", in_.question)
    try:
        answer = _invoke_chain_sync(chain, in_.question, in_.history, in_.top_k)
        # Some chains return dicts; normalize to string (adjust if your chain returns sources)
        if isinstance(answer, dict) and "answer" in answer:
            answer = answer["answer"]
        elif not isinstance(answer, str):
            answer = str(answer)
        return ChatOut(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_stream")
async def chat_stream(in_: ChatIn, chain = Depends(get_chain)):
    """Simple text stream of the model's output tokens.
    Returns a text/plain stream; switch to SSE if your frontend expects it.
    """
    def _gen():
        try:
            stream = getattr(chain, "stream", None)
            if stream is None:
                # Fallback: single-shot
                yield _invoke_chain_sync(chain, in_.question, in_.history, in_.top_k)
                return
            for chunk in stream({
                "question": in_.question,
                "history": [h.model_dump() for h in in_.history],
            }):
                # LangChain Runnable.stream yields strings or dicts
                if isinstance(chunk, dict) and "answer" in chunk:
                    yield chunk["answer"]
                else:
                    yield str(chunk)
        except Exception as e:
            log.exception("Stream error")
            yield f"[STREAM_ERROR] {e}"
    return StreamingResponse(_gen(), media_type="text/plain")
