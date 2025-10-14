
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional, Dict, Any

from langchain_openai import OpenAIEmbeddings
from ..config import settings

__all__ = ["get_embeddings"]


def _openai_kwargs(model_override: Optional[str] = None, dimensions_override: Optional[int] = None) -> Dict[str, Any]:
    """
    Build keyword arguments for OpenAIEmbeddings using project settings + overrides.

    Notes
    -----
    * `text-embedding-3-*` 계열은 `dimensions` 파라미터로 차원을 축소할 수 있습니다.
      (예: 3-large 기본 3072 → 1024/256, 3-small 기본 1536 → 512). 
      실제 지원 여부/제한은 OpenAI 및 LangChain 문서를 참조하세요.
    """
    # Base model (allow override)
    model_name = (model_override or settings.openai_embedding_model).strip()

    # Optional knobs (graceful fallback if not present in settings)
    dims = dimensions_override if dimensions_override is not None else getattr(settings, "openai_embedding_dimensions", None)
    max_retries = getattr(settings, "openai_max_retries", 2)
    request_timeout = getattr(settings, "openai_request_timeout", 60)
    disallowed_special = getattr(settings, "openai_disallowed_special", None)

    # API key precedence: ENV > settings.openai_api_key
    api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)

    kwargs: Dict[str, Any] = {
        "model": model_name,
        # langchain-openai kw
        "max_retries": int(max_retries) if max_retries is not None else 2,
        "request_timeout": float(request_timeout) if request_timeout is not None else 60.0,
    }

    if api_key:
        # langchain-openai expects `api_key` (SecretStr supported); plain str also works
        kwargs["api_key"] = api_key
    if dims:
        # Supported for text-embedding-3-* families
        kwargs["dimensions"] = int(dims)
    if disallowed_special is not None:
        kwargs["disallowed_special"] = disallowed_special

    return kwargs


@lru_cache(maxsize=8)
def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    dimensions: Optional[int] = None,
) -> OpenAIEmbeddings:
    """
    Create (and cache) an embeddings client.

    Parameters
    ----------
    provider : str | None
        Embedding provider name. Defaults to `settings.embedding_provider`.
    model : str | None
        Optional model name override.
    dimensions : int | None
        Optional dimension override for supported models (e.g., text-embedding-3-*).

    Returns
    -------
    OpenAIEmbeddings
        A configured embeddings instance. Instances are memoized per (provider, model, dimensions).
    """
    prov = (provider or settings.embedding_provider).lower().strip()

    if prov == "openai":
        return OpenAIEmbeddings(**_openai_kwargs(model_override=model, dimensions_override=dimensions))

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {prov}. Supported providers: 'openai'"
    )
