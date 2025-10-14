# -*- coding: utf-8 -*-
"""
wandb_sync_openai_ft.py
- OpenAI Fine-tuning job 이벤트/메트릭을 W&B로 백필.
- 입력: --job (ftjob-...) 또는 --model (ft:...) 중 하나.

ENV
  WANDB_API_KEY=...
  WANDB_ENTITY=kordokrip
  WANDB_PROJECT=SVC-FT
  WANDB_RUN_NAME="SFT 41mini backfill"
  WANDB_TAGS="svc,41mini,sft,backfill"
"""
from __future__ import annotations
import os, math, argparse
from typing import Any, Dict, Optional, Tuple
import wandb
from openai import OpenAI

NUM = (int, float)

def _client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION"),
        project=os.getenv("OPENAI_PROJECT"),
    )

def _safe_get(obj, field, default=None):
    try:
        return getattr(obj, field)
    except Exception:
        try:
            d = obj.model_dump()
            return d.get(field, default)
        except Exception:
            return default

def _infer_suffix(job) -> Optional[str]:
    sfx = _safe_get(job, "suffix")
    if sfx:
        return sfx
    ftm = _safe_get(job, "fine_tuned_model")
    if isinstance(ftm, str) and ":" in ftm:
        return ftm.split(":")[-1]  # 마지막 콜론 이후를 접미사로 추정
    return None

def _find_job_id_by_model(client: OpenAI, model: str) -> Optional[str]:
    cursor = None
    while True:
        resp = client.fine_tuning.jobs.list(limit=100, after=cursor)
        for j in resp.data:
            if _safe_get(j, "fine_tuned_model") == model:
                return j.id
        if not getattr(resp, "has_more", False):
            break
        cursor = getattr(resp, "last_id", None)
    return None

def _is_num(x): return isinstance(x, NUM) and not (x != x)  # not NaN

def _extract_metrics(ev: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[int]]:
    d = ev.get("data") or {}
    metrics: Dict[str, float] = {}
    step = d.get("step") if _is_num(d.get("step")) else None
    # 평면 키
    for k in (
        "train_loss", "valid_loss",
        "train_accuracy", "valid_accuracy",
        "train_mean_token_accuracy", "valid_mean_token_accuracy",
        "lr", "epoch", "batch_size", "tokens_processed"
    ):
        v = d.get(k)
        if _is_num(v):
            metrics[k] = float(v)
    # data.metrics 딕셔너리
    m = d.get("metrics")
    if isinstance(m, dict):
        for k, v in m.items():
            if _is_num(v):
                metrics[k] = float(v)
    return metrics, step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="OpenAI job id (ftjob-...)")
    ap.add_argument("--model", help="Fine-tuned model name (ft:...)")
    ap.add_argument("--run-name", help="Override W&B run name")
    ap.add_argument("--tags", help="comma-separated tags")
    args = ap.parse_args()

    if not (args.job or args.model):
        raise SystemExit("Provide --job or --model")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "SVC-FT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=args.run_name or os.getenv("WANDB_RUN_NAME") or "openai-ft-backfill",
        tags=[t.strip() for t in (args.tags or os.getenv("WANDB_TAGS","")).split(",") if t.strip()],
        config={}
    )

    client = _client()

    # job resolve
    job_id = args.job
    if not job_id:
        job_id = _find_job_id_by_model(client, args.model)
        if not job_id:
            run.finish()
            raise SystemExit(f"Could not resolve job id for model={args.model}")

    try:
        job = client.fine_tuning.jobs.retrieve(job_id)

        cfg = {
            "job_id": job.id,
            "status": _safe_get(job, "status"),
            "base_model": _safe_get(job, "model"),
            "output_model": _safe_get(job, "fine_tuned_model"),
            "suffix": _infer_suffix(job),
            "trained_tokens": _safe_get(job, "trained_tokens"),
            "hyperparameters": _safe_get(job, "hyperparameters"),
            "method": _safe_get(job, "method"),
            "training_file": _safe_get(job, "training_file"),
            "validation_file": _safe_get(job, "validation_file"),
            "created_at": _safe_get(job, "created_at"),
        }
        wandb.config.update(cfg, allow_val_change=True)
        wandb.summary["openai_job_url"] = f"https://platform.openai.com/finetune/{job.id}"
        if cfg["output_model"]:
            wandb.summary["openai_model"] = cfg["output_model"]

        # 이벤트 → 메트릭 로깅
        resp = client.fine_tuning.jobs.list_events(job_id, limit=10_000)
        events = getattr(resp, "data", []) or []
        events = sorted(events, key=lambda e: _safe_get(e, "created_at", 0))

        fallback_step = 0
        for ev in events:
            e = ev.model_dump() if hasattr(ev, "model_dump") else dict(ev)
            if e.get("type") not in ("metrics", "message"):
                # 일부 버전은 type 구분 다를 수 있음: metrics 우선
                pass
            metrics, step = _extract_metrics(e)
            if not metrics:
                continue
            if step is None:
                fallback_step += 1
                step = fallback_step
            wandb.log(metrics, step=int(step))

        print("Backfill complete:", job_id, cfg.get("output_model"))
        print("W&B run url:", run.url)
    finally:
        # 예외가 나도 run을 정상 종료시켜 orphan 방지
        try:
            run.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()