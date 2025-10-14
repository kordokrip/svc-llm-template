from __future__ import annotations
import re, argparse, json, hashlib
from pathlib import Path
from rapidfuzz.distance import Cosine

def normalize(txt:str)->str:
    t=re.sub(r"[ \t]+"," ", txt)
    t=re.sub(r"\n{2,}","\n", t)
    return t.strip()

def dedupe(chunks:list[str], sim_threshold=0.95)->list[str]:
    kept=[]; sigs=[]
    for c in chunks:
        n=normalize(c)
        is_dup=False
        for s in sigs:
            if Cosine.normalized_distance(n, s) < (1-sim_threshold):
                is_dup=True; break
        if not is_dup:
            kept.append(n); sigs.append(n)
    return kept

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)    # input txt
    ap.add_argument("--out", dest="out", required=True)   # output txt
    args=ap.parse_args()

    raw=Path(args.inp).read_text(encoding="utf-8")
    chunks=[p for p in re.split(r"\n{2,}", raw) if p.strip()]
    clean=dedupe(chunks)
    Path(args.out).write_text("\n\n".join(clean), encoding="utf-8")
    print(f"[OK] {len(chunks)}→{len(clean)} 청크 (중복제거/정규화)")

if __name__=="__main__":
    main()