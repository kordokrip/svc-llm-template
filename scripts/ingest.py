import argparse
import os, sys
from dotenv import load_dotenv

load_dotenv()

# --- add: src 경로 주입 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# ---------------------------

from svc_llm.rag.loader import load_documents
from svc_llm.rag.splitter import split_docs

# 선택적 의존 (없으면 우회)
try:
    from svc_llm.vectorstore.chroma_store import get_chroma  # type: ignore
except Exception:
    get_chroma = None  # type: ignore
try:
    from svc_llm.vectorstore.pinecone_store import get_pinecone  # type: ignore
except Exception:
    get_pinecone = None  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data", help="Source directory for documents")
    ap.add_argument("--store", choices=["chroma", "pinecone"], default="chroma")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    # 새 인자 (요청 반영)
    ap.add_argument("--persist", default=".chroma", help="Persist directory for Chroma")
    ap.add_argument("--collection", default="default", help="Collection / namespace name")
    ap.add_argument("--emb", default="upstage", help="Embedding model alias (e.g., upstage, openai)")
    args = ap.parse_args()

    # 하위 빌더들이 환경변수로도 받을 수 있게 전파
    os.environ.setdefault("CHROMA_PERSIST_DIR", args.persist)
    os.environ.setdefault("CHROMA_COLLECTION", args.collection)
    os.environ.setdefault("EMBED_MODEL", args.emb)
    os.environ.setdefault("VECTOR_COLLECTION", args.collection)
    os.environ.setdefault("PINECONE_NAMESPACE", args.collection)

    print(f"[ingest] src={args.src} store={args.store} chunk={args.chunk} overlap={args.overlap}")
    if args.store == "chroma":
        print(f"[ingest] chroma.persist={args.persist} chroma.collection={args.collection} emb={args.emb}")
    else:
        print(f"[ingest] pinecone.namespace={args.collection} emb={args.emb}")

    docs = load_documents(args.src)
    chunks = split_docs(docs, chunk_size=args.chunk, chunk_overlap=args.overlap)
    print(f"[ingest] docs={len(docs)} chunks={len(chunks)}")

    if args.store == "chroma":
        vs = None
        if get_chroma is not None:
            # 다양한 시그니처 대응
            try:
                vs = get_chroma(persist_dir=args.persist, collection=args.collection, emb_alias=args.emb)
            except TypeError:
                try:
                    vs = get_chroma(persist_dir=args.persist, collection=args.collection)
                except TypeError:
                    try:
                        vs = get_chroma()
                    except Exception:
                        vs = None
        if vs is None:
            # Fallback: LangChain Chroma 직접 구성
            from langchain_community.vectorstores import Chroma
            # 임베딩 선택
            emb = None
            try:
                alias = (args.emb or "").lower()
                if alias in ("upstage", "korean", "multilingual"):
                    from langchain_upstage import UpstageEmbeddings
                    emb = UpstageEmbeddings(model="solar-embedding-1-large")
                elif alias in ("openai", "oai", "gpt"):
                    from langchain_openai import OpenAIEmbeddings
                    emb = OpenAIEmbeddings()
                else:
                    from langchain_openai import OpenAIEmbeddings
                    emb = OpenAIEmbeddings()
            except Exception as e:
                print(f"[warn] embedding fallback failed: {e}; defaulting to OpenAIEmbeddings")
                from langchain_openai import OpenAIEmbeddings
                emb = OpenAIEmbeddings()
            vs = Chroma(collection_name=args.collection, persist_directory=args.persist, embedding_function=emb)
        vs.add_documents(chunks)
        try:
            vs.persist()
        except Exception:
            pass
        print(f"Chroma에 {len(chunks)}개 청크 저장 완료 → {args.persist} [{args.collection}]")
    else:
        vs = None
        if get_pinecone is not None:
            try:
                vs = get_pinecone(namespace=args.collection, emb_alias=args.emb)
            except TypeError:
                try:
                    vs = get_pinecone()
                except Exception:
                    vs = None
        if vs is None:
            # Fallback: LangChain Pinecone V3 직접 구성
            from langchain_pinecone import PineconeVectorStore
            from langchain_openai import OpenAIEmbeddings
            import pinecone
            emb = OpenAIEmbeddings()
            pc = pinecone.Pinecone()
            index_name = os.environ.get("PINECONE_INDEX") or "svc-index"
            index = pc.Index(index_name)
            vs = PineconeVectorStore(index=index, embedding=emb, namespace=args.collection)
        vs.add_documents(chunks)
        print(f"Pinecone에 {len(chunks)}개 청크 업로드 완료 (namespace={args.collection})")

if __name__ == "__main__":
    main()
