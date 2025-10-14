# scripts/merge_ft_v2.py
from __future__ import annotations
import argparse, glob, io, json, os, random, sys, hashlib
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# --- Optional: tiktoken 사용 (없으면 4 chars≈1 token 근사) ---
def _get_enc():
    try:
        import tiktoken
        for name in ("o200k_base","cl100k_base"):
            try: return tiktoken.get_encoding(name)
            except Exception: pass
    except Exception:
        return None
ENC = _get_enc()

def tok_count_text(s: str) -> int:
    if not s: return 0
    if ENC: return len(ENC.encode(s))
    return max(1, len(s)//4)

def tok_count_messages(msgs: list[dict]) -> int:
    t = 0
    for m in msgs or []:
        c = m.get("content","")
        if isinstance(c,str): t += tok_count_text(c)
    return t

# --- Helpers to coerce heterogeneous fields to text/messages ---
def _to_text(x) -> str:
    """Coerce arbitrary value (str/list/dict/number/None) to a compact text string.
    - str -> stripped
    - list[str|num|dict] -> joined by newlines after str()
    - dict -> json dumps (single-line)
    - None -> ""
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        parts = []
        for it in x:
            if isinstance(it, str):
                parts.append(it.strip())
            else:
                parts.append(json.dumps(it, ensure_ascii=False))
        return "\n".join(p for p in parts if p)
    # dict or number etc.
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _looks_like_messages(lst) -> bool:
    """Return True if lst is a list of dicts having role/content keys."""
    if not isinstance(lst, list):
        return False
    ok = 0
    for m in lst:
        if isinstance(m, dict) and "role" in m and "content" in m:
            ok += 1
    return ok >= max(1, len(lst))  # tolerant (all or most elements)


def _normalize_messages_list(lst: list[dict]) -> list[dict] | None:
    if not isinstance(lst, list) or not lst:
        return None
    norm = []
    for m in lst:
        if not isinstance(m, dict):
            return None
        role = (m.get("role") or "").strip()
        content = m.get("content")
        if role not in ROLE_OK:
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        norm.append({"role": role, "content": content.strip()})
    return norm

# ---------- IO ----------
def read_jsonl(path: str):
    with io.open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try: yield json.loads(line)
                except json.JSONDecodeError: pass

def write_jsonl(path: str, items):
    with io.open(path,"w",encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False)+"\n")

# ---------- Schema check ----------
def is_sft(obj: dict) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("messages"), list)

def is_dpo_pair(obj: dict) -> bool:
    # classic
    if all(k in obj for k in ("prompt","chosen","rejected")): return True
    # input + preferred/non_preferred
    if "input" in obj and ("preferred_output" in obj and "non_preferred_output" in obj): return True
    return False

# ---------- Normalization ----------
ROLE_OK = {"system","user","assistant"}

def normalize_sft(obj: dict, src: str) -> dict|None:
    msgs=obj.get("messages")
    if not isinstance(msgs, list) or not msgs: return None
    norm=[]
    for m in msgs:
        role=(m.get("role") or "").strip()
        content=m.get("content")
        if role not in ROLE_OK: return None
        if not isinstance(content,str) or not content.strip(): return None
        norm.append({"role":role,"content":content.strip()})
    # 최소: user/assistant 각 1회 이상
    if not any(m["role"]=="user" for m in norm): return None
    if not any(m["role"]=="assistant" for m in norm): return None
    out={"messages":norm, "metadata": {"source": os.path.basename(src)}}
    return out

def normalize_dpo(obj: dict, src: str) -> dict|None:
    # case A: prompt/ chosen/ rejected (allow str or list)
    if all(k in obj for k in ("prompt","chosen","rejected")):
        p  = _to_text(obj.get("prompt"))
        ch = _to_text(obj.get("chosen"))
        rj = _to_text(obj.get("rejected"))
        if not (p and ch and rj):
            return None
        return {
            "prompt": p,
            "chosen": ch,
            "rejected": rj,
            "metadata": {"source": os.path.basename(src)}
        }

    # case B: input + preferred/non_preferred (input can be text, messages, list, or dict)
    if "input" in obj and ("preferred_output" in obj and "non_preferred_output" in obj):
        raw_inp = obj.get("input")
        pref = _to_text(obj.get("preferred_output"))
        nonp = _to_text(obj.get("non_preferred_output"))
        if not pref or not nonp:
            return None

        # Normalize input
        inp_norm: str | dict
        if isinstance(raw_inp, dict) and "messages" in raw_inp and isinstance(raw_inp["messages"], list):
            msgs = _normalize_messages_list(raw_inp["messages"])
            if not msgs:
                return None
            inp_norm = {"messages": msgs}
        elif _looks_like_messages(raw_inp):  # list of role/content dicts
            msgs = _normalize_messages_list(raw_inp)
            if not msgs:
                return None
            inp_norm = {"messages": msgs}
        elif isinstance(raw_inp, str):
            if not raw_inp.strip():
                return None
            inp_norm = raw_inp.strip()
        else:
            # any other type -> stringify
            inp_norm = _to_text(raw_inp)
            if not inp_norm:
                return None

        return {
            "input": inp_norm,
            "preferred_output": pref,
            "non_preferred_output": nonp,
            "metadata": {"source": os.path.basename(src)}
        }
    return None

# ---------- Hash (dedup) ----------
def hash_sft(rec: dict) -> str:
    buf=[]
    for m in rec["messages"]:
        buf.append(m["role"]); buf.append("\n"); buf.append(m["content"]); buf.append("\n")
    return hashlib.sha256("".join(buf).encode("utf-8")).hexdigest()

def hash_dpo(rec: dict) -> str:
    if "prompt" in rec:
        key = f"{_to_text(rec.get('prompt'))}\n<<C>>{_to_text(rec.get('chosen'))}\n<<R>>{_to_text(rec.get('rejected'))}"
    else:
        inp = rec.get("input")
        if isinstance(inp, dict):
            inp_key = json.dumps(inp, ensure_ascii=False)
        else:
            inp_key = _to_text(inp)
        key = f"{inp_key}\n<<P>>{_to_text(rec.get('preferred_output'))}\n<<N>>{_to_text(rec.get('non_preferred_output'))}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# ---------- Token count ----------
def tokens_sft(rec: dict) -> int:
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        return tok_count_messages(msgs)
    return 0

def tokens_dpo(rec: dict) -> int:
    if "prompt" in rec:
        return (
            tok_count_text(_to_text(rec.get("prompt"))) +
            tok_count_text(_to_text(rec.get("chosen"))) +
            tok_count_text(_to_text(rec.get("rejected")))
        )
    else:
        inp = rec.get("input")
        if isinstance(inp, dict) and "messages" in inp:
            t_in = tok_count_messages(inp["messages"]) if isinstance(inp["messages"], list) else 0
        elif _looks_like_messages(inp):
            msgs = _normalize_messages_list(inp) or []
            t_in = tok_count_messages(msgs)
        else:
            t_in = tok_count_text(_to_text(inp))
        return t_in + tok_count_text(_to_text(rec.get("preferred_output"))) + tok_count_text(_to_text(rec.get("non_preferred_output")))

# ---------- Split (prompt-group aware) ----------
def split_grouped(items: list[dict], get_group_key, val_ratio: float, rng: random.Random):
    groups=defaultdict(list)
    for it in items:
        groups[get_group_key(it)].append(it)
    keys=list(groups.keys())
    rng.shuffle(keys)
    cut=int(len(keys)*(1.0-val_ratio))
    train_keys=set(keys[:cut])
    train, val = [], []
    for k, arr in groups.items():
        (train if k in train_keys else val).extend(arr)
    return train, val

# ---------- Main merge ----------
def merge(paths: list[str], mode: str, val_ratio: float, seed: int,
          max_train_tokens: int|None, min_tokens: int, max_tokens: int,
          fill_order: str, out_prefix: str):
    rng = random.Random(seed)
    sft, dpo = [], []
    seen_sft, seen_dpo = set(), set()
    raw = {"seen":0,"sft_keep":0,"dpo_keep":0,"sft_drop":0,"dpo_drop":0}

    for p in paths:
        for obj in read_jsonl(p):
            raw["seen"]+=1
            if (mode in ("auto","sft")) and is_sft(obj):
                rec = normalize_sft(obj, p)
                if not rec: raw["sft_drop"]+=1; continue
                t = tokens_sft(rec)
                if t < min_tokens or (max_tokens and t > max_tokens): raw["sft_drop"]+=1; continue
                h = hash_sft(rec)
                if h in seen_sft: continue
                seen_sft.add(h); sft.append(rec); raw["sft_keep"]+=1
            elif (mode in ("auto","dpo")) and is_dpo_pair(obj):
                rec = normalize_dpo(obj, p)
                if not rec: raw["dpo_drop"]+=1; continue
                t = tokens_dpo(rec)
                if t < min_tokens or (max_tokens and t > max_tokens): raw["dpo_drop"]+=1; continue
                h = hash_dpo(rec)
                if h in seen_dpo: continue
                seen_dpo.add(h); dpo.append(rec); raw["dpo_keep"]+=1

    rng.shuffle(sft); rng.shuffle(dpo)

    # prompt-group aware split
    sft_train, sft_val = split_grouped(
        sft,
        get_group_key=lambda r: "|".join(m["content"] for m in r["messages"] if m["role"]=="user")[:256],
        val_ratio=val_ratio, rng=rng
    )
    dpo_train, dpo_val = split_grouped(
        dpo,
        get_group_key=lambda r: (r.get("prompt") or json.dumps(r.get("input"), ensure_ascii=False))[:256],
        val_ratio=val_ratio, rng=rng
    )

    # token budget cut (train only)
    if max_train_tokens:
        used = 0
        kept_sft, kept_dpo = [], []

        if fill_order == "sft-first":
            for r in sft_train:
                t = tokens_sft(r)
                if used + t > max_train_tokens: break
                kept_sft.append(r); used += t
            for r in dpo_train:
                t = tokens_dpo(r)
                if used + t > max_train_tokens: break
                kept_dpo.append(r); used += t
        else:  # dpo-first
            for r in dpo_train:
                t = tokens_dpo(r)
                if used + t > max_train_tokens: break
                kept_dpo.append(r); used += t
            for r in sft_train:
                t = tokens_sft(r)
                if used + t > max_train_tokens: break
                kept_sft.append(r); used += t

        sft_train, dpo_train = kept_sft, kept_dpo

    # save
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    if mode in ("auto","sft"):
        write_jsonl(f"{out_prefix}.sft.train.jsonl", sft_train)
        write_jsonl(f"{out_prefix}.sft.val.jsonl",   sft_val)
    if mode in ("auto","dpo"):
        write_jsonl(f"{out_prefix}.dpo.train.jsonl", dpo_train)
        write_jsonl(f"{out_prefix}.dpo.val.jsonl",   dpo_val)

    # report
    info = {
        "mode": mode, "seed": seed, "val_ratio": val_ratio,
        "max_train_tokens": max_train_tokens, "min_tokens": min_tokens, "max_tokens": max_tokens,
        "counts": {
            "sft": {"train": len(sft_train), "val": len(sft_val)},
            "dpo": {"train": len(dpo_train), "val": len(dpo_val)}
        },
        "train_tokens_est": {
            "sft": sum(tokens_sft(r) for r in sft_train),
            "dpo": sum(tokens_dpo(r) for r in dpo_train)
        },
        "raw_stats": raw,
        "cost_estimates": {
            "gpt-4.1-mini": round((sum(tokens_sft(r) for r in sft_train) + sum(tokens_dpo(r) for r in dpo_train))/1_000_000 * 5, 4)
        }
    }
    with io.open(f"{out_prefix}.dataset_info.json","w",encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(json.dumps(info, ensure_ascii=False, indent=2), file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="글롭 패턴 여러 개 가능")
    ap.add_argument("--mode", choices=["auto","sft","dpo"], default="auto")
    ap.add_argument("--val-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-tokens", type=int, default=None, help="train만 토큰 상한")
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=0, help="0=제한없음")
    ap.add_argument("--fill-order", choices=["sft-first","dpo-first"], default="sft-first")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    paths=[]
    for pat in args.inputs:
        paths.extend(glob.glob(pat))
    if not paths:
        print("입력 파일을 찾지 못했습니다.", file=sys.stderr); sys.exit(1)

    merge(paths, args.mode, args.val_ratio, args.seed,
          args.max_train_tokens, args.min_tokens, args.max_tokens,
          args.fill_order, args.out_prefix)

if __name__=="__main__":
    main()