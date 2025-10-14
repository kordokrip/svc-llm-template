from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Enhanced RAG ingest pipeline (LangChain 0.3.x)
- XLSX → 자연어 문장화(옵션) + 키워드 꼬리표 주입
- 헤더 인지 청킹(MarkdownHeaderTextSplitter) + 길이 기반 분할
- (옵션) SemanticChunker 사용
- 보일러플레이트 제거/간단 중복 제거
- 메타데이터 강화(source/sheet/row/section/tags)
- Chroma(langchain-chroma) 자동 퍼시스트(persist_directory 지정 시)

참고:
- MarkdownHeaderTextSplitter 가이드: https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/
- SemanticChunker: https://python.langchain.com/docs/how_to/semantic-chunker/
- Chroma persist_directory: https://python.langchain.com/docs/integrations/vectorstores/chroma/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
"""

import os
import re
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# 로더들
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

# Optional: SemanticChunker
try:
    from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
    HAS_SEMANTIC = True
except Exception:
    HAS_SEMANTIC = False

# Optional loaders for Excel (Unstructured) and pandas fallback
try:
    from langchain_community.document_loaders.excel import UnstructuredExcelLoader  # type: ignore
    HAS_UNSTRUCTURED_EXCEL = True
except Exception:
    HAS_UNSTRUCTURED_EXCEL = False

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# ---------------------- 유틸/전처리 ----------------------

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(
            f"환경변수 {key} 가 설정되지 않았습니다.\n"
            f"해결: 프로젝트 루트 .env 파일에 아래 예시로 설정 후 재실행하세요.\n"
            f"OPENAI_API_KEY=sk-...\nEMBEDDING_MODEL=text-embedding-3-large\n"
        )
    return val


def strip_boilerplate(text: str) -> str:
    """머리말/꼬리말/페이지 번호 같은 보일러플레이트 간단 제거."""
    text = re.sub(r"\n?Page \d+ of \d+\s*", "\n", text)
    text = re.sub(r"^\s*Confidential\s*$", "", text, flags=re.I | re.M)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


def dedup_by_hash(lines: List[str]) -> List[str]:
    import hashlib
    seen = set(); out: List[str] = []
    for ln in lines:
        h = hashlib.md5(ln.strip().encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); out.append(ln)
    return out


def load_keywords(path: str | None) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] 키워드 파일을 찾지 못했습니다: {path}")
        return []
    raw = p.read_text(encoding="utf-8").strip()
    # CSV/행 단위 혼합 허용: 콤마/공백 기준 토큰화
    toks = []
    for line in raw.splitlines():
        toks.extend([t.strip() for t in re.split(r"[,\s]", line) if t.strip()])
    # 중복 제거
    seen = set(); out = []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def append_keywords_tail(text: str, keywords: List[str]) -> str:
    if not keywords:
        return text
    return text + f"\n\n|| keywords: " + ", ".join(keywords)


# ---------------------- 문서 로딩 ----------------------

def xlsx_to_documents_via_pandas(path: str, keywords: List[str]) -> List[Document]:
    docs: List[Document] = []
    try:
        xls = pd.read_excel(path, sheet_name=None)  # 모든 시트
        for sheet_name, df in xls.items():
            df = df.fillna("")
            for idx, row in df.iterrows():
                parts = []
                for col, val in row.items():
                    if val == "":
                        continue
                    parts.append(f"{col}: {str(val).strip()}")
                if not parts:
                    continue
                sent = " | ".join(parts)
                sent = append_keywords_tail(sent, keywords)
                docs.append(Document(
                    page_content=sent,
                    metadata={
                        "source": path,
                        "sheet": sheet_name,
                        "row": int(idx),
                        "loader": "pandas-row",
                        "lang": "ko",
                    },
                ))
    except Exception as e:
        docs.append(Document(
            page_content=f"PARSE_ERROR_XLSX: {e}",
            metadata={"source": path, "error": True},
        ))
    return docs


def load_docs(src: str, *, xlsx_normalize: bool, keywords: List[str]) -> List[Document]:
    files = [p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True) if os.path.isfile(p)]
    docs: List[Document] = []
    for f in files:
        ext = Path(f).suffix.lower()
        try:
            if ext in {".txt", ".md"}:
                for d in TextLoader(f, encoding="utf-8").load():
                    d.page_content = strip_boilerplate(d.page_content)
                    docs.append(d)
            elif ext == ".pdf":
                docs += PyPDFLoader(f).load()
            elif ext == ".docx":
                # docx2txt는 문단 기준이라 추가 전처리 적용
                tmp = Docx2txtLoader(f).load()
                for d in tmp:
                    d.page_content = strip_boilerplate(d.page_content)
                docs += tmp
            elif ext in {".xlsx", ".xls"}:
                if xlsx_normalize:
                    if HAS_PANDAS:
                        docs += xlsx_to_documents_via_pandas(f, keywords)
                    else:
                        raise ImportError("pandas 가 필요합니다: pip install pandas")
                else:
                    if HAS_UNSTRUCTURED_EXCEL:
                        loader = UnstructuredExcelLoader(f, mode="elements")
                        docs += loader.load()
                    elif HAS_PANDAS:
                        # 시트 전체를 CSV 텍스트로 평탄화
                        xls = pd.read_excel(f, sheet_name=None)
                        for sheet_name, df in xls.items():
                            text = df.to_csv(index=False)
                            docs.append(Document(
                                page_content=text,
                                metadata={"source": f, "sheet": sheet_name, "loader": "pandas"},
                            ))
                    else:
                        raise ImportError("Excel 파서를 사용할 수 없습니다(unstructured/pandas 미설치).")
        except Exception as e:
            docs.append(Document(page_content=f"PARSE_ERROR: {e}", metadata={"source": f, "error": True}))
    return docs


# ---------------------- 청킹 ----------------------

def build_chunks(docs: List[Document], *,
                 chunk: int, overlap: int,
                 semantic: bool,
                 header_aware: bool,
                 embeddings: OpenAIEmbeddings) -> List[Document]:
    """헤더 인지 → 길이 기반 → (옵션) 의미 청킹 전략으로 청크 생성."""
    chunks: List[Document] = []

    # 1) Markdown 헤더 인지 1차 분리 (md/txt에만 의미 있음)
    if header_aware:
        for d in docs:
            # md 헤더가 아니어도 큰 문제 없음(그냥 통과)
            try:
                splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
                sections = splitter.split_text(d.page_content)
                for s in sections:
                    chunks.append(Document(page_content=s.page_content, metadata={**d.metadata, **getattr(s, "metadata", {})}))
            except Exception:
                chunks.append(d)
    else:
        chunks = list(docs)

    # 2) 길이 기반 2차 분리
    char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk,
        chunk_overlap=overlap,
        keep_separator=False,
    )
    size_chunks = char_splitter.split_documents(chunks)

    # 3) (옵션) 의미 청킹으로 미세 조정
    if semantic and HAS_SEMANTIC:
        sem = SemanticChunker(embeddings)
        size_chunks = sem.split_documents(size_chunks)
    elif semantic and not HAS_SEMANTIC:
        print("[WARN] SemanticChunker 미설치 → --semantic-chunk 옵션은 무시됩니다. pip install langchain-experimental")

    # 보일러플레이트 재제거 + 공백 정리
    for d in size_chunks:
        d.page_content = strip_boilerplate(d.page_content)

    return size_chunks


# ---------------------- 메인 ----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data", help="원본 데이터 폴더")
    ap.add_argument("--persist", default=".chroma", help="Chroma 퍼시스트 디렉터리")
    ap.add_argument("--chunk", type=int, default=800, help="chunk size(문자 수)")
    ap.add_argument("--overlap", type=int, default=150, help="chunk overlap")
    ap.add_argument("--min-chars", type=int, default=int(os.getenv("MIN_CHUNK_CHARS", "20")), help="최소 청크 길이")
    ap.add_argument("--header-aware", action="store_true", help="Markdown 헤더 인지 분할 활성화")
    ap.add_argument("--semantic-chunk", action="store_true", help="SemanticChunker 사용")
    ap.add_argument("--xlsx-normalize", action="store_true", help="엑셀을 자연어 문장으로 변환하여 인덱싱")
    ap.add_argument("--keywords", default=None, help="키워드/동의어 파일 경로(쉼표/공백 구분)")
    args = ap.parse_args()

    # 0) 키워드 로딩
    keywords = load_keywords(args.keywords)
    if keywords:
        print(f"[INFO] keywords loaded: {len(keywords)} → e.g., {keywords[:8]}")

    # 1) 로딩
    raw_docs = load_docs(args.src, xlsx_normalize=args.xlsx_normalize, keywords=keywords)

    # 2) 청킹
    require_env("OPENAI_API_KEY")
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    print(f"[INFO] Embedding model = {embed_model}")
    embeddings = OpenAIEmbeddings(model=embed_model)

    chunks = build_chunks(
        raw_docs,
        chunk=args.chunk,
        overlap=args.overlap,
        semantic=args.semantic_chunk,
        header_aware=args.header_aware,
        embeddings=embeddings,
    )

    # 3) 비어있거나 너무 짧은 청크 제거 + 간단 dedup
    before = len(chunks)
    chunks = [d for d in chunks if (d.page_content or "").strip() and len(d.page_content.strip()) >= args.min_chars]
    # 간단 중복 제거(라인 기준)
    if chunks:
        lines = [d.page_content for d in chunks]
        lines = dedup_by_hash(lines)
        # 메타는 잃지만, 동일 청크 중복만 제거하려면 아래처럼 단순 재문서화
        chunks = [Document(page_content=ln, metadata={"dedup": True}) for ln in lines]
    after = len(chunks)

    if after == 0:
        from collections import Counter
        exts = Counter(Path(f).suffix.lower() for f in glob.glob(os.path.join(args.src, "**", "*"), recursive=True) if os.path.isfile(f))
        raise SystemExit(
            "[ERROR] 생성된 청크가 0개입니다. 다음을 점검하세요:\n"
            f"- 지원 포맷(.txt, .md, .pdf, .docx, .xlsx)인지?  소스 폴더='{args.src}'\n"
            "- PDF/Docx 파서 의존성 설치(pypdf, docx2txt) 여부\n"
            "- 원문이 비어있거나 스캔 PDF(텍스트 없음)인지\n"
            "- 필요시 OCR로 텍스트를 먼저 생성하세요.\n"
            f"[참고] 확장자 분포: {dict(exts)}\n"
        )

    print(f"[INFO] Chunks: {before} -> {after} (min_chars={args.min_chars})")
    lens = [len(d.page_content) for d in chunks[:5]]
    print(f"[INFO] sample lengths={lens}")

    # 4) 색인 (Chroma는 persist_directory 지정 시 자동 퍼시스트)
    collection = os.getenv("CHROMA_COLLECTION", "svc-knowledge")
    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=args.persist,
        collection_name=collection,
    )
    print(f"[OK] Chroma 인덱스 생성 완료: {len(chunks)} chunks → dir='{args.persist}', collection='{collection}'")


if __name__ == "__main__":
    main()