from __future__ import annotations
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
import contextvars
from datetime import datetime

# -----------------------------------------------------------------------------
# Context (request/job/user) propagated to every log record via Filter
# -----------------------------------------------------------------------------
_TRACE_ID = contextvars.ContextVar("trace_id", default="-")
_REQUEST_ID = contextvars.ContextVar("request_id", default="-")
_SESSION_ID = contextvars.ContextVar("session_id", default="-")
_USER_ID = contextvars.ContextVar("user_id", default="-")
_JOB_ID = contextvars.ContextVar("job_id", default="-")


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # always True
        record.trace_id = _TRACE_ID.get()
        record.request_id = _REQUEST_ID.get()
        record.session_id = _SESSION_ID.get()
        record.user_id = _USER_ID.get()
        record.job_id = _JOB_ID.get()
        return True


def set_log_context(*, trace_id: Optional[str] = None, request_id: Optional[str] = None,
                    session_id: Optional[str] = None, user_id: Optional[str] = None,
                    job_id: Optional[str] = None) -> None:
    """Set contextual fields that will appear on every log line."""
    if trace_id is not None:
        _TRACE_ID.set(str(trace_id))
    if request_id is not None:
        _REQUEST_ID.set(str(request_id))
    if session_id is not None:
        _SESSION_ID.set(str(session_id))
    if user_id is not None:
        _USER_ID.set(str(user_id))
    if job_id is not None:
        _JOB_ID.set(str(job_id))


# -----------------------------------------------------------------------------
# Formatters (plain / JSON / Rich)
# -----------------------------------------------------------------------------

_DEF_FMT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "trace=%(trace_id)s req=%(request_id)s sess=%(session_id)s user=%(user_id)s job=%(job_id)s | "
    "%(message)s"
)
_DEF_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def _json_formatter():
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore
    except Exception:
        return None
    # Keep keys short to reduce log size; include context keys explicitly.
    fields = [
        "asctime", "levelname", "name", "message",
        "trace_id", "request_id", "session_id", "user_id", "job_id",
        "process", "thread", "module",
    ]
    fmt = " ".join([f"%({k})s" for k in fields])
    jf = jsonlogger.JsonFormatter(fmt=fmt, timestamp=True)
    # Force ISO8601 timestamp
    jf.converter = lambda *a: datetime.now().timetuple()  # type: ignore[attr-defined]
    return jf


def _rich_handler(level: int, use_rich: bool):
    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore
            handler = RichHandler(rich_tracebacks=False, markup=False)
        except Exception:
            handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.addFilter(_ContextFilter())
    return handler


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------

def setup_logging(level: str | int = None,
                  *,
                  json: Optional[bool] = None,
                  rich: Optional[bool] = None,
                  logfile: Optional[str] = None,
                  rotate_mb: int = 20,
                  rotate_backups: int = 3,
                  reset: bool = False) -> logging.Logger:
    """Create an application logger with sensible defaults.

    Env overrides
    -------------
    SVC_LOG_LEVEL=INFO|DEBUG|...   (default INFO)
    SVC_LOG_JSON=true|false        (default false)
    SVC_LOG_RICH=true|false        (default true if TTY)
    SVC_LOG_FILE=path/to/file.log  (optional file logging)

    Idempotent: pass `reset=True` to clear prior handlers.
    """
    # Resolve options from ENV
    env_level = os.getenv("SVC_LOG_LEVEL", "INFO").upper()
    level = level if level is not None else env_level
    level_num = getattr(logging, str(level).upper(), logging.INFO)

    if json is None:
        json = os.getenv("SVC_LOG_JSON", "false").lower() in {"1", "true", "yes"}
    if rich is None:
        rich = os.getenv("SVC_LOG_RICH", "auto").lower() in {"1", "true", "yes", "auto"} and sys.stdout.isatty()
    logfile = logfile or os.getenv("SVC_LOG_FILE")

    root = logging.getLogger()

    # Avoid duplicate handlers / allow full reset
    if reset:
        for h in list(root.handlers):
            root.removeHandler(h)
    elif any(isinstance(h, (logging.StreamHandler, RotatingFileHandler)) for h in root.handlers):
        # Already configured — just return project logger
        return logging.getLogger("svc-llm")

    root.setLevel(level_num)

    # Console handler
    ch = _rich_handler(level_num, use_rich=bool(rich))

    # Formatter selection
    if json:
        jf = _json_formatter()
        if jf is not None:
            ch.setFormatter(jf)
        else:
            ch.setFormatter(logging.Formatter(fmt=_DEF_FMT, datefmt=_DEF_DATEFMT))
    else:
        ch.setFormatter(logging.Formatter(fmt=_DEF_FMT, datefmt=_DEF_DATEFMT))

    root.addHandler(ch)

    # Optional file handler with rotation
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=rotate_mb * 1024 * 1024, backupCount=rotate_backups)
        fh.setLevel(level_num)
        fh.addFilter(_ContextFilter())
        if json and (jfmt := _json_formatter()) is not None:
            fh.setFormatter(jfmt)
        else:
            fh.setFormatter(logging.Formatter(fmt=_DEF_FMT, datefmt=_DEF_DATEFMT))
        root.addHandler(fh)

    # Quiet noisy third‑party loggers by default
    for noisy in ("uvicorn", "uvicorn.access", "httpx", "urllib3", "pinecone", "asyncio"):
        logging.getLogger(noisy).setLevel(max(level_num, logging.WARNING))

    return logging.getLogger("svc-llm")


# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------

def get_logger(name: str = "svc-llm") -> logging.Logger:
    """Return a child logger of the app logger."""
    return logging.getLogger(name)


def harmonize_uvicorn(access_log: bool = False) -> None:
    """Make Uvicorn/FastAPI reuse our logging configuration.

    Tips:
    - Start uvicorn with `log_config=None` to prevent it from overriding logging.
    - You may also disable access logs with `--no-access-log` if not needed.
    """
    # Ensure Uvicorn uses our handlers/levels
    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.error").propagate = True
    logging.getLogger("uvicorn.access").propagate = True
    if not access_log:
        logging.getLogger("uvicorn.access").disabled = True
