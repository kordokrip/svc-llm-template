from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_core.documents import Document

# -----------------------------------------------------------------------------
# KeywordBooster
#  - Query expansion using a domain lexicon (base -> synonyms)
#  - Lightweight reranker that fuses original retrieval score with keyword hits
#  - Safe, fast, and configurable (precompiled regex, caps, quote phrases)
# -----------------------------------------------------------------------------


@dataclass
class BoosterConfig:
    path: str = "data/keywords_svc.json"
    weight: float = 1.2               # contribution of keyword score in rerank
    alpha: float = 1.0                # final_score = orig + alpha * weight * kw_score
    case_insensitive: bool = True
    use_word_boundaries: bool = False # True for strict token matches, False for substring
    max_added_terms: int = 12         # cap expanded terms appended to query
    quote_phrases: bool = True        # wrap multi-word terms in quotes for exact match


class KeywordBooster:
    """Domain keyword booster.

    JSON schema (flexible):
    {
      "스크러버": ["scrubber", "scrubbing"],
      "흡착탑":   ["adsorber", "activated carbon tower"]
    }

    - Keys are *base* terms; values are lists of synonyms/aliases.
    - If the file is missing or malformed, the booster falls back to an empty map.
    """

    def __init__(self, path: str = "data/keywords_svc.json", weight: float = 1.2, *,
                 alpha: float = 1.0,
                 case_insensitive: bool = True,
                 use_word_boundaries: bool = False,
                 max_added_terms: int = 12,
                 quote_phrases: bool = True) -> None:
        self.cfg = BoosterConfig(
            path=path,
            weight=weight,
            alpha=alpha,
            case_insensitive=case_insensitive,
            use_word_boundaries=use_word_boundaries,
            max_added_terms=max_added_terms,
            quote_phrases=quote_phrases,
        )
        self.map: Dict[str, List[str]] = self._load_map(self.cfg.path)
        self._compiled: List[Tuple[re.Pattern, List[re.Pattern], str, List[str]]] = self._compile_patterns(self.map)

    # ------------------------------ public API --------------------------------
    def boost_query(self, q: str) -> str:
        """Expand query with synonyms based on base-term hits in q.
        - Adds up to `max_added_terms` unique terms.
        - Optionally wraps multi-word terms in quotes for exact match.
        """
        q_norm = q or ""
        added: List[str] = []
        seen = set()
        for base_pat, syn_pats, base_str, syns in self._compiled:
            if base_pat.search(q_norm):
                for syn in syns:
                    if syn not in seen:
                        seen.add(syn)
                        added.append(self._format_term(syn))
                        if len(added) >= self.cfg.max_added_terms:
                            break
        if added:
            return f"{q_norm} " + " ".join(sorted(set(added)))
        return q_norm

    def rerank(self, docs: List[Document], q: str) -> List[Document]:
        """Re-rank documents by fusing original score with keyword-hit score.
        - kw_score: count of base+syn hits (case/word-boundary configurable)
        - final_score = orig_score + alpha * weight * kw_score
        - Keeps original docs order as a stable tiebreaker
        - Annotates `doc.metadata['kw_score']` and `['final_score']`
        """
        def orig_score(d: Document) -> float:
            # Try common metadata keys used by various stores
            md = d.metadata or {}
            for k in ("score", "similarity", "relevance", "distance"):
                if k in md and md[k] is not None:
                    try:
                        val = float(md[k])
                        # If this looks like a distance (smaller is better), invert conservatively
                        if k == "distance":
                            return 1.0 - val
                        return val
                    except Exception:
                        continue
            return 0.0

        scored: List[Tuple[float, int, Document]] = []
        for idx, d in enumerate(docs):
            txt = d.page_content or ""
            kw = self._keyword_hits(txt)
            final = orig_score(d) + (self.cfg.alpha * self.cfg.weight * kw)
            d.metadata = dict(d.metadata or {})
            d.metadata.update({"kw_score": kw, "final_score": final})
            scored.append((final, -idx, d))  # -idx to keep stable original order
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [d for _, _, d in scored]

    # ----------------------------- internals ----------------------------------
    @staticmethod
    def _load_map(path: str) -> Dict[str, List[str]]:
        p = Path(path)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Normalize: dict[str, list[str]]
            norm: Dict[str, List[str]] = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        norm[str(k)] = [str(x) for x in v if isinstance(x, (str, int, float))]
                    elif isinstance(v, str):
                        norm[str(k)] = [v]
            return norm
        except Exception:
            return {}

    def _compile_patterns(self, mp: Dict[str, List[str]]):
        flags = re.IGNORECASE if self.cfg.case_insensitive else 0
        compiled: List[Tuple[re.Pattern, List[re.Pattern], str, List[str]]] = []
        for base, syns in mp.items():
            base_pat = self._to_pattern(base, flags)
            syn_pats = [self._to_pattern(s, flags) for s in syns]
            compiled.append((base_pat, syn_pats, base, syns))
        return compiled

    def _to_pattern(self, term: str, flags: int) -> re.Pattern:
        term_escaped = re.escape(term)
        if self.cfg.use_word_boundaries:
            # Use word boundaries; for CJK or hyphenated tokens this may be too strict
            pat = rf"\b{term_escaped}\b"
        else:
            pat = term_escaped
        return re.compile(pat, flags)

    def _format_term(self, term: str) -> str:
        t = str(term).strip()
        if self.cfg.quote_phrases and (" " in t or "\t" in t):
            return f'"{t}"'
        return t

    def _keyword_hits(self, text: str) -> int:
        if not text:
            return 0
        hits = 0
        for base_pat, syn_pats, _, _ in self._compiled:
            if base_pat.search(text):
                hits += 1
            for sp in syn_pats:
                if sp.search(text):
                    hits += 1
        return hits