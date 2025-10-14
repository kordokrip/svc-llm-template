
from __future__ import annotations
import os
import argparse
from typing import Dict, Any, List

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from ..vectorstore.chroma_store import get_chroma
from ..rag.retriever import make_retriever
from ..llm.chain import build_chain

# -----------------------------------------------------------------------------
# LangSmith eval runner (sync). 
# - Dataset은 이름으로 가져오거나, 없으면 생성합니다.
# - chain은 RAG 체인을 그대로 호출하도록 target_fn으로 래핑합니다.
# - evaluators는 필요 시 확장하십시오(정확도/포맷/금칙어 등).
# Docs: evaluate()/datasets/prebuilt evaluators
# -----------------------------------------------------------------------------
# * evaluate docs: https://docs.smith.langchain.com/reference/python/evaluation/langsmith.evaluation._runner.evaluate
# * dataset programmatic mgmt: https://docs.langchain.com/langsmith/manage-datasets-programmatically
# * evaluation quickstart: https://docs.langchain.com/langsmith/evaluation-quickstart
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LangSmith eval against RAG chain")
    p.add_argument("--dataset", default=os.getenv("LS_DATASET", "svc_eval_set"), help="Dataset name")
    p.add_argument("--k", type=int, default=int(os.getenv("SVC_RETRIEVAL_K", "4")), help="Retriever top-k")
    p.add_argument("--exp", default=os.getenv("LS_EXPERIMENT_PREFIX", "svc_rag_eval"), help="Experiment prefix")
    p.add_argument("--limit", type=int, default=int(os.getenv("LS_LIMIT", "0")), help="Optional cap on examples (0=all)")
    return p.parse_args()


def _get_or_create_dataset(client: Client, name: str):
    # Try to read by name; if missing, create
    for ds in client.list_datasets():
        if ds.name == name:
            return ds
    return client.create_dataset(name, description="SVC RAG evaluation dataset")


def main() -> None:
    load_dotenv()
    args = _parse_args()

    client = Client()

    # Build RAG chain
    vs = get_chroma()  # default backend (env로 pinecone 사용 시 API 서버에서 교체 가능)
    retriever = make_retriever(vs, k=args.k)
    chain = build_chain(retriever)

    dataset = _get_or_create_dataset(client, args.dataset)

    # Target function: chain을 직접 호출
    def target_fn(input_: Dict[str, Any]) -> Dict[str, Any]:
        q = input_["question"]
        # chain.invoke는 문자열 또는 {"answer": str}을 리턴하도록 구성
        out = chain.invoke({"question": q, "history": []})
        if isinstance(out, dict) and "answer" in out:
            return {"answer": out["answer"]}
        return {"answer": str(out)}

    # NOTE: 필요 시 evaluator를 추가하세요. (예: Exact match, string distance, LLM-as-a-judge 등)
    evaluators: List[Any] = []

    # 대규모 데이터셋일 때는 aevaluate() + max_concurrency 설정 권장
    results = evaluate(
        target=target_fn,
        data=dataset if args.limit <= 0 else list(client.list_examples(dataset_id=dataset.id))[: args.limit],
        evaluators=evaluators,
        experiment_prefix=args.exp,
    )
    print("✅ Done. Experiment:", results)


if __name__ == "__main__":
    main()
