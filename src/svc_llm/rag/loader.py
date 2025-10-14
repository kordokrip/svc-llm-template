from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Iterable, Dict

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)

# -----------------------------------------------------------------------------
# Robust multi-format loader for RAG ingestion
#  - Safe encoding handling for text/markdown
#  - PDF loads per page via PyPDFLoader
#  - DOC/DOCX via Unstructured (fallback: docx2txt)
#  - CSV/XLSX via pandas with row caps to prevent token explosion
#  - Skips hidden/system dirs and very large files
# -----------------------------------------------------------------------------

# Limits (override via ENV)
CSV_ROW_LIMIT = int(os.getenv("RAG_CSV_ROW_LIMIT", "5000"))
XLSX_ROW_LIMIT = int(os.getenv("RAG_XLSX_ROW_LIMIT", "3000"))
MAX_FILE_MB = int(os.getenv("RAG_MAX_FILE_MB", "50"))  # skip files bigger than this

SKIP_DIRS = {".git", ".chroma", ".venv", "__pycache__", "node_modules"}


def _is_hidden_or_skipped(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    # macOS hidden files or dotfiles
    if any(p.startswith(".") for p in path.parts if p not in {".", ".."}):
        return True
    return False


def _too_big(path: Path) -> bool:
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        return size_mb > MAX_FILE_MB
    except Exception:
        return False


# ------------------------------ TXT / MD -------------------------------------

def _load_txt_md(root: Path) -> List[Document]:
    docs: List[Document] = []
    for ext in ("*.txt", "*.md"):
        for f in root.rglob(ext):
            if not f.is_file() or _is_hidden_or_skipped(f) or _too_big(f):
                continue
            try:
                # autodetect_encoding=True가 실패 시 내부적으로 재시도 수행
                loader = TextLoader(str(f), autodetect_encoding=True)
                for d in loader.load():
                    d.metadata = {**(d.metadata or {}), "source": str(f), "type": f.suffix.lstrip("."), "loader": "text"}
                    docs.append(d)
            except Exception as e:
                docs.append(Document(page_content=f"TEXT parse error: {e}", metadata={"source": str(f), "type": f.suffix.lstrip("."), "loader": "text", "error": True}))
    return docs


# ------------------------------- PDF -----------------------------------------

def _load_pdf(root: Path) -> List[Document]:
    docs: List[Document] = []
    for f in root.rglob("*.pdf"):
        if not f.is_file() or _is_hidden_or_skipped(f) or _too_big(f):
            continue
        try:
            loader = PyPDFLoader(str(f))
            for d in loader.load():  # returns page-level Documents
                d.metadata = {**(d.metadata or {}), "source": str(f), "type": "pdf", "loader": "pypdf"}
                docs.append(d)
        except Exception as e:
            docs.append(Document(page_content=f"PDF parse error: {e}", metadata={"source": str(f), "type": "pdf", "loader": "pypdf", "error": True}))
    return docs


# ------------------------------- DOC / DOCX ----------------------------------

def _load_docx(root: Path) -> List[Document]:
    docs: List[Document] = []
    for pattern in ("*.docx", "*.doc"):
        for f in root.rglob(pattern):
            if not f.is_file() or _is_hidden_or_skipped(f) or _too_big(f):
                continue
            # 1st try: Unstructured (더 풍부한 분해가 가능)
            try:
                loader = UnstructuredWordDocumentLoader(str(f), mode="single")
                for d in loader.load():
                    d.metadata = {**(d.metadata or {}), "source": str(f), "type": f.suffix.lstrip("."), "loader": "unstructured"}
                    docs.append(d)
                continue
            except Exception:
                pass
            # Fallback: docx2txt
            try:
                for d in Docx2txtLoader(str(f)).load():
                    d.metadata = {**(d.metadata or {}), "source": str(f), "type": f.suffix.lstrip("."), "loader": "docx2txt"}
                    docs.append(d)
            except Exception as e:
                docs.append(Document(page_content=f"DOCX parse error: {e}", metadata={"source": str(f), "type": f.suffix.lstrip("."), "loader": "docx2txt", "error": True}))
    return docs


# ------------------------------- CSV -----------------------------------------

def _load_csv(root: Path) -> List[Document]:
    docs: List[Document] = []
    for f in root.rglob("*.csv"):
        if not f.is_file() or _is_hidden_or_skipped(f) or _too_big(f):
            continue
        try:
            df = pd.read_csv(str(f), low_memory=False)
            if CSV_ROW_LIMIT and len(df) > CSV_ROW_LIMIT:
                df = df.head(CSV_ROW_LIMIT)
            text = df.to_csv(index=False)
            docs.append(Document(page_content=text, metadata={"source": str(f), "type": "csv", "rows": len(df)}))
        except Exception as e:
            docs.append(Document(page_content=f"CSV parse error: {e}", metadata={"source": str(f), "type": "csv", "error": True}))
    return docs


# ------------------------------- XLS/XLSX ------------------------------------

def _load_xls_xlsx(root: Path) -> List[Document]:
    docs: List[Document] = []
    for pattern in ("*.xlsx", "*.xlsm", "*.xls"):
        for f in root.rglob(pattern):
            if not f.is_file() or _is_hidden_or_skipped(f) or _too_big(f):
                continue
            try:
                xl = pd.ExcelFile(str(f))  # engine auto-detect; openpyxl recommended for xlsx
                for sheet in xl.sheet_names:
                    try:
                        df = xl.parse(sheet)
                        if XLSX_ROW_LIMIT and len(df) > XLSX_ROW_LIMIT:
                            df = df.head(XLSX_ROW_LIMIT)
                        text = df.to_csv(index=False)
                        docs.append(Document(page_content=text, metadata={"source": str(f), "sheet": sheet, "type": "xlsx", "rows": len(df)}))
                    except Exception as se:
                        docs.append(Document(page_content=f"XLSX sheet parse error: {se}", metadata={"source": str(f), "sheet": sheet, "type": "xlsx", "error": True}))
            except Exception as e:
                docs.append(Document(page_content=f"XLSX open error: {e}", metadata={"source": str(f), "type": "xlsx", "error": True}))
    return docs


# --------------------------------- PUBLIC ------------------------------------

def load_documents(root_dir: str, *, include: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from a directory recursively with sensible defaults.

    Parameters
    ----------
    root_dir : str
        Root directory to scan.
    include : list[str] | None
        File types to include. Defaults to ["txt","md","pdf","docx","doc","csv","xlsx","xlsm","xls"].

    Returns
    -------
    list[Document]
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    allow = set([e.lower().lstrip(".") for e in (include or ["txt", "md", "pdf", "docx", "doc", "csv", "xlsx", "xlsm", "xls"])])

    docs: List[Document] = []

    if {"txt", "md"} & allow:
        docs.extend(_load_txt_md(root))
    if {"pdf"} & allow:
        docs.extend(_load_pdf(root))
    if {"docx", "doc"} & allow:
        docs.extend(_load_docx(root))
    if {"csv"} & allow:
        docs.extend(_load_csv(root))
    if {"xlsx", "xlsm", "xls"} & allow:
        docs.extend(_load_xls_xlsx(root))

    return docs