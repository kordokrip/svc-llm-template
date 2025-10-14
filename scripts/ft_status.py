# scripts/ft_status.py
from __future__ import annotations
import argparse, os, sys, time, json
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _ts(i: Optional[int]) -> str:
    if not i: return "-"
    try:
        return datetime.fromtimestamp(i, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        return str(i)

def _pp(o):
    print(json.dumps(o, ensure_ascii=False, indent=2))

def list_jobs(client: OpenAI, limit: int, model_filter: Optional[str], raw: bool):
    items = client.fine_tuning.jobs.list(limit=limit).data
    # 최신순
    items.sort(key=lambda x: x.created_at or 0, reverse=True)
    if model_filter:
        mf = model_filter.strip()
        items = [j for j in items if (j.model and mf in j.model) or (j.fine_tuned_model and mf in j.fine_tuned_model)]
    if raw:
        _pp([j.to_dict() for j in items])
        return
    # 요약 테이블
    print(f"{'JOB_ID':<32} {'MODEL':<28} {'STATUS':<12} {'FT_MODEL':<40} {'CREATED':<20} {'FINISHED':<20}")
    for j in items:
        print(f"{j.id:<32} {str(j.model)[:28]:<28} {str(j.status)[:12]:<12} {str(j.fine_tuned_model)[:40]:<40} {_ts(j.created_at):<20} {_ts(getattr(j,'finished_at',None)):<20}")

def show_job(client: OpenAI, job_id: str, raw: bool, events: int):
    job = client.fine_tuning.jobs.retrieve(job_id)
    evs = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=events).data if events>0 else []
    if raw:
        out = job.to_dict()
        out["_events"] = [e.to_dict() for e in evs]
        _pp(out); return
    # 사람친화 요약
    print("=== JOB ===")
    print("id:", job.id)
    print("status:", job.status)
    print("base model:", job.model)
    print("fine_tuned_model:", job.fine_tuned_model)
    print("created_at:", _ts(job.created_at))
    if hasattr(job, "finished_at"):
        print("finished_at:", _ts(getattr(job, "finished_at")))
    if getattr(job, "hyperparameters", None):
        print("hyperparameters:", job.hyperparameters)
    if getattr(job, "training_file", None):
        print("training_file:", job.training_file)
    if getattr(job, "validation_file", None):
        print("validation_file:", job.validation_file)
    if getattr(job, "result_files", None):
        print("result_files:", job.result_files)

    if evs:
        print("\n=== RECENT EVENTS ===")
        for e in reversed(evs):
            line = f"[{_ts(e.created_at)}] {e.type}"
            md = getattr(e, "data", None) or getattr(e, "metadata", None)
            if md:
                try:
                    if isinstance(md, dict):
                        # 대표 지표 몇 개만 노출
                        keys = ["train_loss", "valid_loss", "accuracy", "mean_token_accuracy", "learning_rate"]
                        pairs = [f"{k}={md[k]}" for k in keys if k in md]
                        if pairs:
                            line += "  " + ", ".join(pairs)
                except Exception:
                    pass
            print(line)

def watch_job(client: OpenAI, job_id: str, interval: float):
    seen = set()
    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            evs = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50).data
            for e in reversed(evs):
                if e.id in seen: 
                    continue
                seen.add(e.id)
                line = f"[{_ts(e.created_at)}] {e.type}"
                md = getattr(e, "data", None) or getattr(e, "metadata", None)
                if md and isinstance(md, dict):
                    if "message" in md:
                        line += f" :: {md['message']}"
                    # 흔한 메트릭
                    for k in ("train_loss","valid_loss","learning_rate","mean_token_accuracy"):
                        if k in md: line += f"  {k}={md[k]}"
                print(line)
            if job.status in ("succeeded","failed","cancelled"):
                print(f"\n[END] job status = {job.status}")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[STOP] watching cancelled by user.")

def main():
    ap = argparse.ArgumentParser(description="OpenAI fine-tuning job status helper")
    ap.add_argument("--list", type=int, default=0, help="List recent N jobs")
    ap.add_argument("--model", default=os.getenv("FT_BASE_MODEL","gpt-4.1-mini-2025-04-14"), help="Filter jobs by model substring")
    ap.add_argument("--job", help="Show one job by id")
    ap.add_argument("--events", type=int, default=10, help="Number of recent events to show with --job")
    ap.add_argument("--watch", action="store_true", help="Watch events (requires --job)")
    ap.add_argument("--interval", type=float, default=5.0, help="Watch poll interval (sec)")
    ap.add_argument("--json", action="store_true", help="Raw JSON output")
    args = ap.parse_args()

    client = OpenAI()

    if args.list>0:
        list_jobs(client, limit=args.list, model_filter=args.model, raw=args.json)
        return

    if args.job:
        if args.watch:
            watch_job(client, args.job, args.interval)
        else:
            show_job(client, args.job, raw=args.json, events=args.events)
        return

    ap.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)