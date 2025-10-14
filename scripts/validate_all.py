#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preflight validator for SFT/DPO jsonl files before OpenAI fine-tuning.

- SFT/DPO 스키마 검사 (OpenAI 대시보드 요구 형식과 일치)
- DPO에 한해 오프라인 안전성 스캔(정책 휴리스틱)
- o200k_base 토크나이저 기준 추정 토큰 수(미설치/오류 시 러프 추정)
- '--budget-tokens'로 학습 토큰 예산(대략)을 체크 (파일명에 'train' 포함 파일만 예산 합산)
- 모든 파일을 검사하고, 하나라도 invalid/unsafe이면 종료코드 2로 실패 반환

사용:
  python scripts/validate_all.py --root data/finetune --budget-tokens 8000000
"""

import argparse, json, re, sys
from pathlib import Path
from collections import Counter

# ---- token counter (o200k_base 우선) ----
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
except Exception:
    _enc = None

def tok_count(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # fallback: 아주 러프한 추정
    return max(1, len(text) // 4)

# ---- I/O ----
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except Exception as e:
                # 파싱 에러도 리포트
                yield ln, {"__parse_error__": str(e)}

# ---- message helpers ----
def is_msg(d):
    return (
        isinstance(d, dict)
        and d.get("role") in ("system", "user", "assistant")
        and isinstance(d.get("content"), str)
        and d["content"].strip() != ""
    )

def is_assistant_msg(d):
    return (
        isinstance(d, dict)
        and d.get("role") == "assistant"
        and isinstance(d.get("content"), str)
        and d["content"].strip() != ""
    )

def detect_kind(obj):
    if isinstance(obj, dict) and "input" in obj and "preferred_output" in obj and "non_preferred_output" in obj:
        return "dpo"
    if isinstance(obj, dict) and "messages" in obj:
        return "sft"
    return "unknown"

# ---- schema validators ----
def validate_sft(obj):
    errs = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        errs.append("missing or empty 'messages'")
        return errs
    for i, m in enumerate(msgs, 1):
        if not is_msg(m):
            errs.append(f"messages[{i}] must be object with role & string content")
    return errs

def validate_dpo_schema(obj):
    errs = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]

    inp = obj.get("input")
    if not isinstance(inp, dict):
        errs.append("missing key 'input' (object)")
        return errs

    msgs = inp.get("messages")
    if not isinstance(msgs, list) or not msgs:
        errs.append("input.messages must be array[message]")
    else:
        for i, m in enumerate(msgs, 1):
            if not is_msg(m):
                errs.append(f"input.messages[{i}] invalid (must have role & string content)")

    pref = obj.get("preferred_output")
    npref = obj.get("non_preferred_output")

    if not isinstance(pref, list) or not pref:
        errs.append("preferred_output must be array[assistant_message]")
    else:
        for i, m in enumerate(pref, 1):
            if not is_assistant_msg(m):
                errs.append(f"preferred_output[{i}] must be assistant message object")

    if not isinstance(npref, list) or not npref:
        errs.append("non_preferred_output must be array[assistant_message]")
    else:
        for i, m in enumerate(npref, 1):
            if not is_assistant_msg(m):
                errs.append(f"non_preferred_output[{i}] must be assistant message object")

    return errs

# ---- offline safety patterns (convert_dpo_v2.py와 동일 계열) ----
_REFUSAL_FRAGMENTS = [
    # EN
    "I can't help with that", "I cannot assist with", "I won't provide",
    "I will not provide", "I can't provide", "I can't assist",
    "against policy", "illegal", "unsafe",
    # KO
    "도와드릴 수 없습니다", "제공할 수 없습니다", "해당 요청은 불가", "정책에 위배", "불법", "위험합니다",
]

_PATTERNS = {
    "sexual_minors": [
        r"\b(child|children|minor|teen|youth)\b.*\b(sex|porn|explicit)\b",
        r"(아동|미성년|청소년).*(성|음란)",
    ],
    "sexual_explicit": [
        r"\b(porn|pornography|xxx|nsfw)\b", r"(포르노|음란물|노골적 성)"
    ],
    "weapons": [
        r"\b(make|build|assemble|manufacture)\b.*\b(bomb|explosive|weapon|gunpowder)\b",
        r"(폭탄|폭약|무기).*(제조|만드는 법|조립)",
    ],
    "drugs": [
        r"\b(meth|heroin|cocaine|lsd|mdma|fentanyl)\b.*\b(make|synth|cook)\b",
        r"(마약|펜타닐|헤로인|메스암페타민|엑스터시).*(제조|합성|만드는 법)",
    ],
    "cybercrime": [
        r"\b(hack|sql injection|keylogger|ransomware|ddos|crack)\b",
        r"(해킹|랜섬웨어|피싱|스피어피싱|디도스|크랙)"
    ],
    "self_harm": [
        r"\b(kill myself|suicide|self-harm)\b",
        r"(자살|극단적 선택|자해).*(방법|하는 법)"
    ],
    "violence": [
        r"\b(kill|assassinate|murder)\b.*\b(how to|guide|manual|steps)\b",
        r"(살인|암살|폭행).*(방법|하는 법)"
    ],
    "extremism": [
        r"\b(isis|islamic state|al[- ]qaeda|taliban|nazis?)\b.*\b(praise|support|glorify)\b",
        r"(나치|히틀러|백인\s*우월|이슬람국가|탈레반).*(찬양|지지|옹호)"
    ],
    "secrets": [
        r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"ASIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}", r"ghp_[0-9A-Za-z]{36}",
        r"xox[abp]-[0-9A-Za-z-]{10,}", r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----",
    ],
    "pii": [
        r"\b\d{6}-\d{7}\b",   # KR RRN
        r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
    ],
}

def any_match(text: str, pats) -> bool:
    if not text:
        return False
    for p in pats:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False

def contains_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(frag.lower() in t for frag in _REFUSAL_FRAGMENTS)

def safety_scan_dpo(rec):
    msgs = rec.get("input", {}).get("messages", [])
    itext = "\n".join([m.get("content", "") for m in msgs if isinstance(m, dict)])
    ptext = "\n".join([m.get("content", "") for m in rec.get("preferred_output", []) if isinstance(m, dict)])
    ntext = "\n".join([m.get("content", "") for m in rec.get("non_preferred_output", []) if isinstance(m, dict)])

    hit = []
    # block-anywhere classes
    for k in ("sexual_minors", "secrets"):
        if any_match(itext, _PATTERNS[k]) or any_match(ptext, _PATTERNS[k]) or any_match(ntext, _PATTERNS[k]):
            hit.append(k)
            return hit  # immediate block

    risky = []
    for k in ("sexual_explicit","weapons","drugs","cybercrime","self_harm","violence","extremism","pii"):
        if any_match(itext, _PATTERNS[k]) or any_match(ptext, _PATTERNS[k]):
            risky.append(k)

    if risky and not contains_refusal(ptext):
        hit.extend(risky)
        return hit

    # preferred_output가 해로운 가이던스를 주면 차단
    for k in ("weapons","drugs","cybercrime","violence","self_harm"):
        if any_match(ptext, _PATTERNS[k]) and not contains_refusal(ptext):
            hit.append(k)
            return hit

    return hit  # empty -> safe

# ---- per-file validator ----
def validate_file(path: Path):
    kind_counts = Counter()
    invalid_rows = 0
    unsafe_rows = 0
    unsafe_cat = Counter()
    total_tokens = 0

    for ln, obj in read_jsonl(str(path)):
        if "__parse_error__" in obj:
            print(f"[{path}:{ln}] PARSE ERROR -> {obj['__parse_error__']}")
            invalid_rows += 1
            continue

        kind = detect_kind(obj)
        if kind == "sft":
            errs = validate_sft(obj)
            texts = [m.get("content", "") for m in obj.get("messages", []) if isinstance(m, dict)]
        elif kind == "dpo":
            errs = validate_dpo_schema(obj)
            texts = []
            for m in obj.get("input", {}).get("messages", []) or []:
                if isinstance(m, dict):
                    texts.append(m.get("content", ""))
            for m in obj.get("preferred_output", []) or []:
                if isinstance(m, dict):
                    texts.append(m.get("content", ""))
            for m in obj.get("non_preferred_output", []) or []:
                if isinstance(m, dict):
                    texts.append(m.get("content", ""))
        else:
            errs = ["cannot detect dataset kind (expect sft or dpo)"]
            texts = []

        if errs:
            print(f"[{path}:{ln}] {kind} INVALID -> {errs}")
            invalid_rows += 1
            if invalid_rows >= 100:
                break

        if kind == "dpo" and not errs:
            hits = safety_scan_dpo(obj)
            if hits:
                unsafe_rows += 1
                for k in hits:
                    unsafe_cat[k] += 1

        total_tokens += sum(tok_count(t) for t in texts)
        kind_counts[kind] += 1

    print(f"[REPORT] {path} -> lines={sum(kind_counts.values())}, invalid={invalid_rows}, unsafe={unsafe_rows}, tokens≈{total_tokens}")
    if unsafe_cat:
        print(f"[UNSAFE BREAKDOWN] {dict(unsafe_cat)}")

    return {
        "ok": invalid_rows == 0 and unsafe_rows == 0,
        "tokens": total_tokens,
        "invalid": invalid_rows,
        "unsafe": unsafe_rows,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/finetune")
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="학습용(train) 파일들의 총 추정 토큰 상한. 파일명에 'train'이 포함된 jsonl만 합산.")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        print(f"[WARN] no jsonl files found under {root}")
        sys.exit(1)

    ok = True
    total_train_tokens = 0

    for p in files:
        res = validate_file(p)
        if "train" in p.name.lower():
            total_train_tokens += res["tokens"]
        if not res["ok"]:
            ok = False

    if args.budget_tokens is not None:
        print(f"[BUDGET] train tokens≈{total_train_tokens} / cap={args.budget_tokens}")
        if total_train_tokens > args.budget_tokens:
            print("[BUDGET] Exceeded token cap. Trim or sample your train files.")
            ok = False

    if ok:
        print("\n✅ PRE-FLIGHT PASS — safe & schema-valid. Ready to submit.")
        sys.exit(0)
    else:
        print("\n❌ PRE-FLIGHT FAIL — fix the issues above before submitting.")
        sys.exit(2)

if __name__ == "__main__":
    main()