#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cap_dpo_tokens.py
- DPO v3 jsonl에서 누적 토큰이 MAX 토큰을 넘지 않도록 앞에서부터 샘플링
- 토큰 근사: len(text)//4 (4자 ≈ 1토큰 가정)
"""
from __future__ import annotations
import json, argparse
from dotenv import load_dotenv

load_dotenv()

def approx_tokens(s:str)->int:
    return max(1, len(s)//4)

def rec_tokens(rec:dict)->int:
    t=0
    for m in rec["input"]["messages"]:
        t+=approx_tokens(m.get("content",""))
    for m in rec["preferred_output"]:
        t+=approx_tokens(m.get("content",""))
    for m in rec["non_preferred_output"]:
        t+=approx_tokens(m.get("content",""))
    return t

def cap(in_path,out_path,max_tokens:int):
    n=kept=tokens=0
    with open(in_path,"r",encoding="utf-8") as fi, open(out_path,"w",encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): continue
            rec=json.loads(line)
            rt=rec_tokens(rec)
            if tokens+rt>max_tokens: break
            fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
            tokens+=rt; kept+=1
            n+=1
    print(f"[CAPPED] {kept} examples, ~{tokens:,} tokens -> {out_path}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args=ap.parse_args()
    cap(args.input, args.output, args.max_tokens)

if __name__=="__main__":
    main()