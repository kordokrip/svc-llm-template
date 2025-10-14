from __future__ import annotations
"""
Pinecone 업로드 스크립트 (LangChain 0.3+ / pinecone>=3 / langchain-pinecone>=0.2)
- .env 로딩, 인덱스 자동 생성/대기, 확장자별 로더(txt/md/pdf/docx + 선택적 xlsx 정규화)
- OpenAIEmbeddings(text-embedding-3-*) 차원 자동 매핑(3-large=3072, 3-small=1536)
- 서버리스 인덱스 생성 시 cloud/region 인자 사용 (예: aws/us-east-1)
참고: PineconeVectorStore 사용법 / 인덱스 ready 대기 로직은 공식 문서 예제를 따름.
"""

import os
import time
import glob
import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# 유틸: 임베딩 차원 추론
# -----------------------------
EMBED_DIM_MAP = {
    "text-embedding-3-large": 3072,  # OpenAI 공식(3072)
    "text-embedding-3-small": 1536,
}


def infer_dim(model: str) -> int:
    model = (model or "").strip()
    for k, v in EMBED_DIM_MAP.items():
        if k in model:
            return v
    # 기타 모델은 기본 1536으로 가정
    return 1536


# -----------------------------
# 유틸: XLSX → 텍스트 정규화(선택)
# -----------------------------

def _xlsx_to_docs(path: str) -> List[Document]:
    """xlsx를 판다스로 읽어 시트/열 단위로 텍스트화. 숫자/날짜 깨짐 최소화.
    - 셀 값이 비어있거나 공백만인 경우 스킵
    - 시트별로 메타데이터(sheet, source) 부여
    """
    import pandas as pd  # 지연 임포트

    docs: List[Document] = []
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        return [Document(page_content=f"PARSE_ERROR[xlsx]: {e}", metadata={"source": path, "error": True})]

    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name, dtype=str).fillna("")
            # 열 헤더를 보존하며 레코드 단위 텍스트 생성
            rows = []
            for _, row in df.iterrows():
                parts = []
                for col, val in row.items():
                    val = str(val).strip()
                    if val:
                        parts.append(f"{col}: {val}")
                if parts:
                    rows.append(" \n".join(parts))
            if rows:
                content = f"[#sheet: {sheet_name}]\n" + "\n\n".join(rows)
                docs.append(Document(page_content=content, metadata={"source": path, "sheet": sheet_name}))
        except Exception as e:
            docs.append(Document(page_content=f"PARSE_ERROR[xlsx:{sheet_name}]: {e}", metadata={"source": path, "sheet": sheet_name, "error": True}))
    return docs


# -----------------------------
# 로더: 폴더 내 문서 일괄 로드
# -----------------------------

def load_docs(src: str, include_xlsx: bool = False) -> List[Document]:
    files = [p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True) if os.path.isfile(p)]
    docs: List[Document] = []
    for f in files:
        ext = Path(f).suffix.lower()
        try:
            if ext in {".txt", ".md"}:
                docs += TextLoader(f, encoding="utf-8").load()
            elif ext == ".pdf":
                docs += PyPDFLoader(f).load()
            elif ext == ".docx":
                docs += Docx2txtLoader(f).load()
            elif include_xlsx and ext == ".xlsx":
                docs += _xlsx_to_docs(f)
            else:
                # 미지원 확장자는 스킵(메타만 남기고 싶다면 아래 주석 해제)
                # docs.append(Document(page_content=f"SKIP: unsupported {ext}", metadata={"source": f}))
                pass
        except Exception as e:
            docs.append(Document(page_content=f"PARSE_ERROR: {e}", metadata={"source": f, "error": True}))
    return docs


# -----------------------------
# Pinecone 인덱스 보장 및 ready 대기
# -----------------------------

def ensure_index(name: str, dim: int, cloud: str, region: str) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # 키 없으면 즉시 에러 발생시켜 빠르게 실패
    # 현재 프로젝트의 인덱스 목록 조회
    existing = [idx_info["name"] for idx_info in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    # ready 대기
    while True:
        status = pc.describe_index(name).status
        if bool(status.get("ready")):
            break
        time.sleep(1)


# -----------------------------
# 메인
# -----------------------------

def main():
    load_dotenv(find_dotenv(usecwd=True), override=False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data", help="소스 폴더")
    ap.add_argument("--index", default=os.getenv("PINECONE_INDEX", "svc-knowledge"))
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE"))
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--xlsx-normalize", action="store_true", help=".xlsx를 텍스트로 정규화하여 포함")
    ap.add_argument("--pool-threads", type=int, default=4, help="업서트 쓰레드 수")
    args = ap.parse_args()

    # 필수 키 체크
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY 가 설정되어 있지 않습니다.")
    if not os.getenv("PINECONE_API_KEY"):
        raise SystemExit("[ERROR] PINECONE_API_KEY 가 설정되어 있지 않습니다.")

    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embed = OpenAIEmbeddings(model=embed_model)
    dim = infer_dim(embed_model)

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    # 인덱스 생성 및 ready 대기
    ensure_index(args.index, dim=dim, cloud=cloud, region=region)

    # 문서 적재 및 청크 분할
    docs = load_docs(args.src, include_xlsx=args.xlsx_normalize)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk, chunk_overlap=args.overlap)
    splits = splitter.split_documents(docs)

    if not splits:
        print("[WARN] 생성된 청크가 없습니다. 지원 포맷 또는 --xlsx-normalize 옵션을 확인하세요.")
        return

    # Pinecone 업로드
    vs = PineconeVectorStore.from_documents(
        splits,
        embedding=embed,
        index_name=args.index,
        namespace=args.namespace,
        pool_threads=args.pool_threads,
    )

    print(f"[OK] Pinecone 업로드 완료: {len(splits)} chunks → index='{args.index}' namespace='{args.namespace}' (model={embed_model}, dim={dim})")


if __name__ == "__main__":
    main()