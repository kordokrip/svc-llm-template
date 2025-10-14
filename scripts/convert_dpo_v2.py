#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_dpo_v2.py — DPO/SFT normalizer + (optional) safety filtering

Usage examples
--------------
# Convert to OpenAI DPO schema + strict offline safety filter + scan report
python scripts/convert_dpo_v2.py --mode dpo \
  --in  data/finetune/dpo_standard.train.clean.jsonl \
  --out data/finetune/dpo_standard.train.v5.safe.jsonl \
  --policy-filter --strict --scan-report data/finetune/dpo_scan_train_v5.json

# (Optional) also call OpenAI moderation for each record (requires OPENAI_API_KEY)
python scripts/convert_dpo_v2.py --mode dpo \
  --in  data/finetune/dpo_standard.train.clean.jsonl \
  --out data/finetune/dpo_standard.train.v5.safe.jsonl \
  --policy-filter --online-moderation --scan-report data/finetune/dpo_scan_train_v5.json

# SFT cleanup (messages normalization only)
python scripts/convert_dpo_v2.py --mode sft \
  --in  data/finetune/svc_merged.sft.train.clean.jsonl \
  --out data/finetune/svc_merged.sft.train.v5.jsonl
"""

import argparse, json, random, os, re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---- Token counter (o200k_base preferred) ----
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")  # 4.1/4o 계열 추천
except Exception:
    _enc = None

def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)

# ---- Optional online moderation (omni-moderation-latest) ----
_use_online_moderation = False
try:
    from openai import OpenAI
    def _moderate_online(text: str):
        if not text:
            return False, {}
        client = OpenAI()  # picks up OPENAI_API_KEY / ORG / PROJECT if set
        try:
            res = client.moderations.create(model="omni-moderation-latest", input=text)
            r = res.results[0]
            flagged = bool(getattr(r, "flagged", False))
            cats = {}
            if hasattr(r, "categories") and r.categories:
                cats = {k: bool(v) for k, v in r.categories.items()}
            return flagged, cats
        except Exception:
            return False, {}
except Exception:
    def _moderate_online(text: str):
        return False, {}

# ---- Helpers ----

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except Exception as e:
                raise ValueError(f"[{path}:{ln}] JSON parse error: {e}")

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- Content normalization ----

def _to_str_from_content(content):
    if isinstance(content, list):
        buf = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                buf.append(str(p.get("text", "")))
            elif isinstance(p, str):
                buf.append(p)
        return "".join(buf).strip()
    if isinstance(content, dict):
        return (content.get("text") or content.get("content") or json.dumps(content, ensure_ascii=False)).strip()
    return (content or "").strip()

ALLOWED_ROLES = {"system", "user", "assistant"}

def _norm_assistant_messages(value):
    msgs = []
    if value is None:
        return msgs
    if isinstance(value, str):
        s = value.strip()
        if s:
            msgs.append({"role": "assistant", "content": s})
        return msgs
    if isinstance(value, dict):
        content = _to_str_from_content(value.get("content"))
        if content:
            msgs.append({"role": "assistant", "content": content})
        return msgs
    if isinstance(value, list):
        for it in value:
            msgs.extend(_norm_assistant_messages(it))
        return msgs
    s = str(value).strip()
    if s:
        msgs.append({"role": "assistant", "content": s})
    return msgs

def _norm_messages_array(arr):
    out = []
    if not isinstance(arr, list):
        return out
    for m in arr:
        if isinstance(m, dict) and m.get("role") in ALLOWED_ROLES:
            content = _to_str_from_content(m.get("content"))
            if content:
                out.append({"role": m["role"], "content": content})
    return out

def _norm_user_messages(value, default_system=None):
    # value could be: {messages:[...]}, list[message], or raw str
    if isinstance(value, dict) and "messages" in value:
        msgs = _norm_messages_array(value["messages"])
        return {"messages": msgs} if msgs else None
    if isinstance(value, list):
        msgs = _norm_messages_array(value)
        return {"messages": msgs} if msgs else None
    if isinstance(value, str):
        s = value.strip()
        if s:
            msgs = []
            if default_system:
                msgs.append({"role": "system", "content": default_system})
            msgs.append({"role": "user", "content": s})
            return {"messages": msgs}
    return None

# ---- Safety patterns (expanded) ----
_REFUSAL_FRAGMENTS = [
    # EN
    "I can't help with that","I cannot assist with","I won't provide","I will not provide",
    "I can't provide","I can't assist","against policy","illegal","unsafe",
    # KO
    "도와드릴 수 없습니다","제공할 수 없습니다","해당 요청은 불가","정책에 위배","불법","위험합니다",
]

_PATTERNS = {
    # BLOCK ALWAYS
    "sexual_minors": [r"\b(child|children|minor|teen|youth)\b.*\b(sex|porn|explicit)\b", r"(아동|미성년|청소년).*(성|음란)"],
    "bestiality": [r"(sex|sexual|porn).*(animal|beastial|bestial)", r"(수간)"],
    "secrets": [r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}", r"ASIA[0-9A-Z]{16}", r"AIza[0-9A-Za-z\-_]{35}", r"ghp_[0-9A-Za-z]{36}", r"xox[abp]-[0-9A-Za-z-]{10,}", r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"],

    # RISKY (drop by default; allow keep only with --allow-risky-refusals)
    "sexual_explicit": [r"\b(porn|pornography|xxx|nsfw)\b", r"(포르노|음란물|노골적 성)"],
    "sexual_general": [r"\b(sex|sexual|erotic)\b", r"(성행위|에로|야한)"],
    "weapons": [r"\b(make|build|assemble|manufacture)\b.*\b(bomb|explosive|weapon|gunpowder|ghost gun)\b", r"(폭탄|폭약|무기|유령총).*(제조|만드는 법|조립)"],
    "drugs": [r"\b(meth|heroin|cocaine|lsd|mdma|fentanyl|anabolic)\b.*\b(make|synth|cook|brew)\b", r"(마약|펜타닐|헤로인|메스암페타민|엑스터시|스테로이드).*(제조|합성|만드는 법|양조)"],
    "cybercrime": [r"\b(hack|sql injection|keylogger|ransomware|ddos|crack|0day)\b", r"(해킹|랜섬웨어|피싱|스피어피싱|디도스|크랙|제로데이)"],
    "self_harm": [r"\b(kill myself|suicide|self-harm)\b", r"(자살|극단적 선택|자해).*(방법|하는 법)"],
    "violence": [r"\b(kill|assassinate|murder)\b.*\b(how to|guide|manual|steps)\b", r"(살인|암살|폭행).*(방법|하는 법)"],
    "extremism": [r"\b(isis|islamic state|al[- ]qaeda|taliban|nazis?)\b.*\b(praise|support|glorify)\b", r"(나치|히틀러|백인\s*우월|이슬람국가|탈레반).*(찬양|지지|옹호)"],
    "hate": [r"\b(genocide|ethnic cleansing)\b", r"(말살|민족청소)"],
    "harassment": [r"\b(kill yourself|kys|go die)\b", r"(죽어버려|꺼져|자살해)"],
    "crime": [r"\b(shoplift|pickpocket|steal|fraud|money laundering)\b", r"(소매치기|절도|사기|자금세탁)"] ,
    "pii": [r"\b\d{6}-\d{7}\b", r"\b\d{3}-\d{2}-\d{4}\b"],
}

def _any_match(text: str, patterns) -> bool:
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def _contains_refusal(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(frag.lower() in t for frag in _REFUSAL_FRAGMENTS)

# Safety policy: conservative by default
# - ALWAYS blocks: sexual_minors, bestiality, secrets
# - By default also DROP risky categories even if the preferred answer is a refusal.
#   You can allow keeping some “risky but refused” with --allow-risky-refusals (still filtered by ratio)

def policy_filter(rec, stats: Counter, allow_risky_refusals=False):
    msgs = rec.get("input", {}).get("messages", [])
    itext = "\n".join([m.get("content", "") for m in msgs if isinstance(m, dict)])
    ptext = "\n".join([m.get("content", "") for m in rec.get("preferred_output", []) if isinstance(m, dict)])
    ntext = "\n".join([m.get("content", "") for m in rec.get("non_preferred_output", []) if isinstance(m, dict)])

    # Block-anywhere
    for k in ("sexual_minors","bestiality","secrets"):
        pats = _PATTERNS[k]
        if _any_match(itext, pats) or _any_match(ptext, pats) or _any_match(ntext, pats):
            stats[k] += 1
            return False, [k]

    risky_hit = []
    for k in ("sexual_explicit","sexual_general","weapons","drugs","cybercrime","self_harm","violence","extremism","hate","harassment","crime","pii"):
        pats = _PATTERNS[k]
        if _any_match(itext, pats) or _any_match(ptext, pats) or _any_match(ntext, pats):
            risky_hit.append(k)

    if not risky_hit:
        return True, []

    if not allow_risky_refusals:
        for k in risky_hit:
            stats[k] += 1
        return False, risky_hit

    # allow only if preferred is explicit refusal (for those kept)
    if not _contains_refusal(ptext):
        for k in risky_hit:
            stats[k] += 1
        return False, risky_hit

    return True, risky_hit

# ---- Converters ----

def convert_to_openai_dpo(rows, default_system=None, max_train_tokens=None, seed=42,
                           apply_policy_filter=False, allow_risky_refusals=False,
                           online_moderation=False, scan_report=None, risky_ratio_cap=0.01):
    out = []
    total_tokens = 0
    stats = Counter()
    risky_kept = 0
    random.seed(seed)

    for _, obj in rows:
        # Build input
        inp = obj.get("input")
        if inp is None and "messages" in obj:
            inp = {"messages": obj["messages"]}
        if inp is None and "prompt" in obj:
            inp = obj["prompt"]
        inp_norm = _norm_user_messages(inp, default_system=default_system)
        if not inp_norm or not inp_norm.get("messages"):
            # try fallback from 'user' or 'question'
            fallback = obj.get("user") or obj.get("question") or ""
            inp_norm = _norm_user_messages(fallback, default_system=default_system)
        if not inp_norm or not inp_norm.get("messages"):
            continue

        pref = _norm_assistant_messages(obj.get("preferred_output") or obj.get("chosen") or obj.get("preferred"))
        npref = _norm_assistant_messages(obj.get("non_preferred_output") or obj.get("rejected") or obj.get("non_preferred"))
        if not pref or not npref:
            continue

        rec = {"input": inp_norm, "preferred_output": pref, "non_preferred_output": npref}

        # Offline policy filter
        if apply_policy_filter:
            keep, reasons = policy_filter(rec, stats, allow_risky_refusals=allow_risky_refusals)
            if not keep:
                continue
            if reasons:
                risky_kept += 1
                # Cap fraction of risky-kept examples
                if len(out) > 0 and (risky_kept / (len(out) + 1)) > risky_ratio_cap:
                    # drop to keep dataset conservative
                    continue

        # Optional online moderation
        if online_moderation:
            bundle_text = []
            for m in rec["input"]["messages"]:
                bundle_text.append(m["content"])
            for m in rec["preferred_output"]:
                bundle_text.append(m["content"])
            for m in rec["non_preferred_output"]:
                bundle_text.append(m["content"])
            flagged, cats = _moderate_online("\n".join(bundle_text))
            if flagged:
                stats["online_flagged"] += 1
                continue

        # Token budget
        if max_train_tokens is not None:
            texts = []
            for m in rec["input"]["messages"]:
                texts.append(m["content"])
            for m in rec["preferred_output"]:
                texts.append(m["content"])
            for m in rec["non_preferred_output"]:
                texts.append(m["content"])
            toks = sum(count_tokens(t) for t in texts)
            if total_tokens + toks > max_train_tokens:
                continue
            total_tokens += toks

        out.append(rec)

    # Report
    if scan_report:
        try:
            rep = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "examples": len(out),
                "approx_tokens": total_tokens,
                "category_counts": dict(stats),
                "risky_kept": risky_kept,
            }
            with open(scan_report, "w", encoding="utf-8") as rf:
                json.dump(rep, rf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return out, total_tokens


def clean_sft(rows):
    cleaned = []
    for _, obj in rows:
        msgs = obj.get("messages")
        if not isinstance(msgs, list):
            continue
        norm = _norm_messages_array(msgs)
        if norm:
            cleaned.append({"messages": norm})
    return cleaned

# ---- CLI ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dpo","sft"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="input jsonl path")
    ap.add_argument("--out", dest="out", required=True, help="output jsonl path")
    ap.add_argument("--system", default=None, help="default system prompt to inject")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="token budget for train set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy-filter", action="store_true", help="apply offline safety filter")
    ap.add_argument("--allow-risky-refusals", action="store_true", help="keep risky prompts only if preferred is explicit refusal (still ratio-capped)")
    ap.add_argument("--risky-ratio-cap", type=float, default=0.01, help="max fraction of risky-but-kept examples (default 1%)")
    ap.add_argument("--online-moderation", action="store_true", help="call omni-moderation-latest (requires OPENAI_API_KEY)")
    ap.add_argument("--scan-report", default=None, help="write JSON scan summary")
    args = ap.parse_args()

    rows = list(read_jsonl(args.inp))

    if args.mode == "dpo":
        out, total = convert_to_openai_dpo(
            rows,
            default_system=args.system,
            max_train_tokens=args.max_train_tokens,
            seed=args.seed,
            apply_policy_filter=args.policy_filter,
            allow_risky_refusals=args.allow_risky_refusals,
            online_moderation=args.online_moderation,
            scan_report=args.scan_report,
            risky_ratio_cap=max(0.0, min(0.2, args.risky_ratio_cap)),
        )
        write_jsonl(args.out, out)
        print(f"[OK] wrote {len(out)} -> {args.out} (approx tokens: {total})")
    else:
        out = clean_sft(rows)
        write_jsonl(args.out, out)
        print(f"[OK] wrote {len(out)} -> {args.out}")

if __name__ == "__main__":
    main()