#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, io, json, argparse, time, hashlib
from typing import Any, Dict, Iterable, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------- helpers -----------------

def _read_first_n(path: str, n: int = 3) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{i} JSON decode error: {e}")
            if len(out) >= n:
                break
    if not out:
        raise ValueError(f"{path} is empty or unreadable")
    return out

VALID_ROLES = {"system","user","assistant"}

def _is_messages(lst) -> bool:
    return isinstance(lst, list) and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in lst
    )

# ---- schema validation (fast, conservative) ----

def validate_sft_file(path: str) -> None:
    rows = _read_first_n(path, 5)
    for i, r in enumerate(rows, 1):
        if list(r.keys()) != ["messages"]:
            raise ValueError(f"SFT schema error at {path} (example {i}): only 'messages' key is allowed; found {list(r.keys())}")
        if not _is_messages(r["messages"]):
            raise ValueError(f"SFT schema error at {path} (example {i}): messages must be list of {{role,content}}")
        for m in r["messages"]:
            if m["role"] not in VALID_ROLES:
                raise ValueError(f"SFT schema error at {path} (example {i}): invalid role {m['role']}")
            if not isinstance(m["content"], str) or not m["content"].strip():
                raise ValueError(f"SFT schema error at {path} (example {i}): empty content")


def _as_text_list(x: Any) -> List[str]:
    if x is None:
        return []
    try:
        if isinstance(x, str):
            # try JSON-decoding maybe '{"role":"assistant",...}'
            try:
                obj = json.loads(x)
                if isinstance(obj, dict) and "content" in obj:
                    return [str(obj["content"]).strip()] if str(obj["content"]).strip() else []
            except Exception:
                pass
            return [x.strip()] if x.strip() else []
        if isinstance(x, (int, float)):
            return [str(x)]
        if isinstance(x, dict):
            if "content" in x:
                s = str(x["content"]).strip()
                return [s] if s else []
            return [json.dumps(x, ensure_ascii=False)]
        if isinstance(x, (list, tuple)):
            out = []
            for it in x:
                out.extend(_as_text_list(it))
            return [s for s in out if s]
    except Exception:
        return []
    return []


def validate_dpo_file(path: str) -> None:
    rows = _read_first_n(path, 5)
    for i, r in enumerate(rows, 1):
        if "input" not in r:
            raise ValueError(f"DPO schema error at {path} (example {i}): missing key 'input'")
        inp = r["input"]
        if isinstance(inp, dict):
            if not _is_messages(inp.get("messages")):
                raise ValueError(f"DPO schema error at {path} (example {i}): input.messages must be list of {{role,content}}")
        elif not isinstance(inp, str):
            raise ValueError(f"DPO schema error at {path} (example {i}): input must be string or {{messages:[...]}}")
        pref = _as_text_list(r.get("preferred_output"))
        nonp = _as_text_list(r.get("non_preferred_output"))
        if not isinstance(r.get("preferred_output"), list):
            raise ValueError(f"DPO schema error at {path} (example {i}): preferred_output must be array[str]")
        if not isinstance(r.get("non_preferred_output"), list):
            raise ValueError(f"DPO schema error at {path} (example {i}): non_preferred_output must be array[str]")
        if not pref:
            raise ValueError(f"DPO schema error at {path} (example {i}): preferred_output empty after normalization")
        if not nonp:
            raise ValueError(f"DPO schema error at {path} (example {i}): non_preferred_output empty after normalization")

# ---- OpenAI helpers ----

def _upload(client: OpenAI, path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        file = client.files.create(file=f, purpose="fine-tune")
    return file.id


def _integrations_from_env() -> Optional[List[Dict[str, Any]]]:
    # Server-side integration. No wandb SDK import required.
    proj = os.getenv("WANDB_PROJECT")
    ent  = os.getenv("WANDB_ENTITY")
    name = os.getenv("WANDB_RUN_NAME")
    tags = [t for t in (os.getenv("WANDB_TAGS","")) .split(",") if t]
    if not proj:
        return None
    wb = {"project": proj}
    if ent: wb["entity"] = ent
    if name: wb["name"] = name
    if tags: wb["tags"] = tags
    return [{"type":"wandb", "wandb": wb}]

# ----------------- main flow -----------------

def run_jobs(
    model: str,
    sft_train: Optional[str], sft_val: Optional[str],
    dpo_train: Optional[str], dpo_val: Optional[str],
    n_epochs: int, batch: Any, lr_mult: Any, seed: Optional[int],
    run_suffix: str
):
    client = OpenAI()
    integrations = _integrations_from_env()

    sft_job_id = None
    dpo_job_id = None

    if sft_train:
        print("[SFT] Validating…", sft_train)
        validate_sft_file(sft_train)
        if sft_val:
            validate_sft_file(sft_val)
        tr_id = _upload(client, sft_train)
        va_id = _upload(client, sft_val) if sft_val else None
        print("[SFT] Files:", tr_id, va_id)

        print("[SFT] Submitting job…")
        sft_hp = {"n_epochs": n_epochs, "batch_size": batch, "learning_rate_multiplier": lr_mult}
        if seed is not None:
            sft_hp["seed"] = seed
        sft_job = client.fine_tuning.jobs.create(
            model=model,
            training_file=tr_id,
            validation_file=va_id,
            hyperparameters=sft_hp,
            suffix=f"{run_suffix}-SFT",
            integrations=integrations,
        )
        sft_job_id = sft_job.id
        print("[SFT] Job ID:", sft_job_id)

    if dpo_train:
        print("[DPO] Validating…", dpo_train)
        validate_dpo_file(dpo_train)
        if dpo_val:
            validate_dpo_file(dpo_val)
        tr_id = _upload(client, dpo_train)
        va_id = _upload(client, dpo_val) if dpo_val else None
        print("[DPO] Files:", tr_id, va_id)

        print("[DPO] Submitting job…")
        dpo_hp = {"n_epochs": n_epochs, "batch_size": batch, "learning_rate_multiplier": lr_mult, "beta": "auto"}
        dpo_job = client.fine_tuning.jobs.create(
            model=model,
            training_file=tr_id,
            validation_file=va_id,
            method={"type":"dpo", "dpo": {"hyperparameters": dpo_hp}},
            suffix=f"{run_suffix}-DPO",
            integrations=integrations,
        )
        dpo_job_id = dpo_job.id
        print("[DPO] Job ID:", dpo_job_id)

    print("\n=== SUBMITTED ===")
    if sft_job_id: print("SFT:", sft_job_id)
    if dpo_job_id: print("DPO:", dpo_job_id)
    if integrations:
        print("W&B integration enabled:", integrations[0]["wandb"])  # shows project/name/tags
    print("Use: python scripts/ft_status.py --job <ID> --watch --interval 5")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft-train")
    ap.add_argument("--sft-val")
    ap.add_argument("--dpo-train")
    ap.add_argument("--dpo-val")
    ap.add_argument("--epochs", type=int, default=int(os.getenv("FT_EPOCHS", "1")))
    ap.add_argument("--batch", default=os.getenv("FT_BATCH", "auto"))
    ap.add_argument("--lr-mult", default=os.getenv("FT_LR_MULT", "auto"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("FT_SEED", "0")))
    ap.add_argument("--suffix", default=os.getenv("FT_SUFFIX", "SVC-41mini-SFT-DPO-<=80USD"))
    args = ap.parse_args()

    run_jobs(
        model=args.model,
        sft_train=args.sft_train, sft_val=args.sft_val,
        dpo_train=args.dpo_train, dpo_val=args.dpo_val,
        n_epochs=args.epochs, batch=args.batch, lr_mult=args.lr_mult,
        seed=args.seed, run_suffix=args.suffix,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)