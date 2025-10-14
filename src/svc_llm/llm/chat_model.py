
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env for local development convenience
load_dotenv()

_DEFAULT_MODEL = (
    os.getenv("SVC_MODEL")
    or os.getenv("OPENAI_CHAT_MODEL")
    or "gpt-4.1-mini-2025-04-14"
)
_DEFAULT_TEMP = float(os.getenv("SVC_TEMPERATURE", "0.2"))
_DEFAULT_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", os.getenv("SVC_REQUEST_TIMEOUT", "60")))
_DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))


@lru_cache(maxsize=8)
def _get_chat_cached(
    model: str,
    temperature: float,
    streaming: bool,
    timeout: float,
    max_retries: int,
) -> ChatOpenAI:
    """Memoized ChatOpenAI factory (hashable args only).
    See ChatOpenAI params: model, temperature, request_timeout (alias timeout), max_retries, streaming.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        request_timeout=timeout,  # alias of `timeout` in LangChain
        max_retries=max_retries,
        streaming=streaming,
    )


def get_chat(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = True,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    use_cache: bool = True,
) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance.

    Parameters
    ----------
    model : Optional[str]
        Overrides default chat model. Defaults to env SVC_MODEL → OPENAI_CHAT_MODEL → gpt-4.1-mini-2025-04-14.
    temperature : Optional[float]
        Sampling temperature. Defaults to env SVC_TEMPERATURE (0.2).
    streaming : bool
        Enable token streaming. Defaults to True (Runnable.stream() 지원).
    timeout : Optional[float]
        Request timeout (seconds). Defaults to env OPENAI_REQUEST_TIMEOUT (or SVC_REQUEST_TIMEOUT) → 60.
    max_retries : Optional[int]
        Number of retries. Defaults to env OPENAI_MAX_RETRIES → 2.
    use_cache : bool
        If True, cache instances per (model, temperature, streaming, timeout, max_retries).
    """
    m = (model or _DEFAULT_MODEL).strip()
    t = float(_DEFAULT_TEMP if temperature is None else temperature)
    to = float(_DEFAULT_TIMEOUT if timeout is None else timeout)
    r = int(_DEFAULT_MAX_RETRIES if max_retries is None else max_retries)

    if use_cache:
        return _get_chat_cached(m, t, streaming, to, r)
    return ChatOpenAI(model=m, temperature=t, request_timeout=to, max_retries=r, streaming=streaming)
