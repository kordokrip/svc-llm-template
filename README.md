# SVC LLM App Template (Optimized & Refactored)

본 템플릿은 LangChain + Streamlit + FastAPI 기반의 **SVC AI 채팅 애플리케이션**을
즉시 실행/평가/파인튜닝할 수 있도록 최적화한 구조입니다. 실제 사내 문서 포맷 혼재(.txt/.md/.pdf/.docx/.doc/.csv/.xlsx/.pptx/.ppt)를 **자동 분기 로더**로 안전 처리하며, Chroma→Pinecone 이전을 전제로 설계되었습니다.

---

## ⚡ 빠른 시작 (한 줄씩 실행)

```bash
# 0) Conda
conda create -n langchain-basics python=3.12.11 -y && conda activate langchain-basics

# 1) 필수 패키지 설치
pip install -r requirements.txt

# (선택) 1-1) 프로젝트를 editable로 설치하면 Import 문제가 영구 해결
pip install -e .

# 2) 환경변수 템플릿 복사
cp .env.example .env
# .env 에 OPENAI_API_KEY (필수), LangSmith/W&B/Pinecone (선택) 입력

# 3) (선택) .doc/.ppt도 처리하려면 unstructured 추가
pip install "unstructured>=0.15.0"

# 4) 데이터 인덱싱 (로컬 Chroma)
python scripts/ingest.py --src data --store chroma
#  → "Chroma에 N개 청크 저장 완료" 메시지 확인
#  (Pinecone로 업로드하려면 --store pinecone)

# 5) API 서버 (FastAPI)
uvicorn svc_llm.api.main:app --host 0.0.0.0 --port 8000

# 6) UI (Streamlit) - 다른 터미널에서
API_URL=http://127.0.0.1:8000/chat \
  streamlit run src/svc_llm/ui/ChatUI.py --server.port 8501 --server.address 0.0.0.0

# 7) LangSmith 추적/평가 활성 (셸 환경변수)
export LANGSMITH_TRACING=true; export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=...; export LANGSMITH_PROJECT=SVC-LLM
```

> **참고**: editable 설치를 하지 않았다면 임시로 `PYTHONPATH=$(pwd)/src`를 앞에 붙여 실행해도 됩니다.

---

## 🧩 지원 포맷 & 로더 전략
- **텍스트/마크다운**: `TextLoader` (.txt/.md)
- **PDF**: `PyPDFLoader` (.pdf)
- **Word**: `Docx2txtLoader` (.docx) → 실패 시 `unstructured`로 폴백 / **.doc**은 unstructured 필요
- **CSV/XLSX**: `pandas` 로 테이블을 텍스트로 변환하여 문맥화
- **PowerPoint**: `unstructured` 필요 (.ppt/.pptx)
- 실패 파일은 `metadata.error=True`로 Document에 남겨 사후 점검 가능

---

## 📚 디렉토리 구조

```
svc-llm-template/
  .env.example
  requirements.txt
  README.md
  data/                       # 사내 문서
  scripts/
    ingest.py                 # 문서 → 벡터 인덱싱(Chroma/Pinecone)
    ft_sft.py                 # SFT 업로드/학습
    ft_dpo.py                 # DPO 업로드/학습
    ft_status.py              # 파인튜닝 작업 상태 조회
  src/
    svc_llm/
      __init__.py
      config.py               # 환경설정(Pydantic Settings, extras=ignore)
      logging.py              # 구조적 로깅 설정
      embeddings/
        provider.py           # 임베딩 프로바이더 (OpenAI 기본)
      vectorstore/
        chroma_store.py       # 로컬 Chroma
        pinecone_store.py     # Pinecone (선택)
      rag/
        splitter.py           # 청킹/오버랩
        loader.py             # 파일 로더(확장 포맷 자동 분기)
        retriever.py          # 리트리버 구성
      llm/
        chat_model.py         # OpenAI LLM 래퍼
        prompts.py            # 시스템/사용자 프롬프트
        chain.py              # RAG Chain 정의
      api/
        main.py               # FastAPI 엔드포인트(/chat)
      ui/
        ChatUI.py             # Streamlit UI (ChatGPT 유사)
      eval/
        run_langsmith_eval.py # LangSmith 평가 러너
        datasets/             # 평가셋(JSONL 등)
  tests/
    test_chain_smoke.py       # 최소 스모크 테스트
```

---

## 🔐 .env 가이드 (필수/선택)
필수
```
OPENAI_API_KEY=sk-...
```
선택
```
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=SVC-LLM

PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=svc-knowledge

WANDB_API_KEY=...
WANDB_PROJECT=SVC-FT
```
> 본 템플릿의 `config.py`는 pydantic v2 설정으로 **정의되지 않은 키를 무시(extra=ignore)** 합니다.

---

## 🏁 SFT 업로드 & 학습 (gpt-4o-2024-11-20)

### 1) 학습 데이터(JSONL) 형식
- **SFT**: 각 줄이 `{"messages": [{"role":"system|user|assistant", "content":"..."}, ...]}` 형태
- 예시(한 줄):
```json
{"messages":[
  {"role":"system","content":"당신은 SVC 전문가입니다."},
  {"role":"user","content":"자재 KIT 교체 절차 알려줘"},
  {"role":"assistant","content":"1) 준비물 확인 2) 안전조치 ..."}
]}
```

> 준비된 예시 파일: `/mnt/data/ft_sft_budget30_usd.jsonl` (환경에 따라 다른 경로를 사용해도 됩니다)

### 2) 실행
```bash
# (선택) W&B 연동으로 지표 기록
export WANDB_API_KEY=...; export WANDB_PROJECT=SVC-FT

# 기본 경로(/mnt/data/ft_sft_budget30_usd.jsonl)를 사용할 경우
python scripts/ft_sft.py

# 직접 경로를 지정하고 싶다면
SFT_JSONL=/absolute/path/to/train.jsonl python scripts/ft_sft.py
```
- 성공 시 콘솔에 `SFT job id: ftjob_...` 형식의 ID가 출력됩니다.
- 작업 상태 확인:
```bash
python scripts/ft_status.py <ftjob_id>
```

### 3) 모델 사용
- SFT 완료 후 OpenAI 콘솔/응답의 **파인튜닝 모델 ID**를 확인합니다. 
- 앱에서 해당 모델을 쓰려면 `src/svc_llm/llm/chat_model.py`의 기본값을 교체하거나, 호출부에서 `get_chat(model="<fine-tuned-id>")`로 지정하세요.

> DPO(선호학습)도 동일한 절차로 `scripts/ft_dpo.py`를 실행하면 됩니다. 경로 오버라이드는 `DPO_JSONL=/path/file.jsonl`.

---

## 📈 LangSmith 평가 러너
```bash
export LANGSMITH_TRACING=true; export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=...; export LANGSMITH_PROJECT=SVC-LLM
python src/svc_llm/eval/run_langsmith_eval.py
```
- `client.create_dataset(...)`로 만든 평가셋에 대해 체인을 실행하고 지표를 기록합니다. 필요 시 LLM-as-a-judge, 유사도, 포맷 검증 평가자를 추가하세요.

---

## 🧪 로컬 점검 스니펫
```bash
# Settings가 .env를 잘 읽는지
python - <<'PY'
from svc_llm.config import settings
print("OK:", settings.langsmith_tracing, settings.langchain_tracing_v2, settings.wandb_project)
PY

# 로더만 단독 실행해 문서 개수 확인
python - <<'PY'
from svc_llm.rag.loader import load_documents
print("loaded:", len(load_documents("data")))
PY
```

---

## 🩺 자주 묻는 오류(트러블슈팅)
- **`ModuleNotFoundError: svc_llm`** → `pip install -e .`(권장) 또는 `PYTHONPATH=$(pwd)/src`로 실행.
- **Pydantic `ValidationError: Extra inputs are not permitted`** → 템플릿의 `config.py`는 이미 `extra=ignore`. 구버전이라면 `SettingsConfigDict(extra="ignore")`로 패치.
- **`Docx2txtLoader` 임포트 에러** → `from langchain_community.document_loaders import Docx2txtLoader` 사용, `pip install docx2txt` 확인.
- **`.doc`/`.ppt` 미지원** → `pip install unstructured` 설치 또는 해당 파일을 .docx/.pptx로 변환 후 투입.
- **PDF/Excel 파싱 문제** → `pip install pypdf`/`pandas` 확인. 실패 파일은 `metadata.error=True`로 로그에 남습니다.

---

## 📦 수동 설치(요구사항 파일 없이)
```bash
pip install "openai>=1.40" "langchain>=0.2" langchain-community langchain-openai langchain-text-splitters \
  "langchain-pinecone>=0.1" chromadb "pinecone-client>=5" fastapi "uvicorn[standard]" streamlit tiktoken \
  pydantic-settings python-dotenv wandb "pandas>=2" numpy tenacity httpx rich "langsmith>=0.1.98" \
  docx2txt pypdf
```

---

## 운영 팁
- **Chroma → Pinecone 이전**: 로컬에서 품질 고정 후 운영 인덱스(Pinecone)로 이전.
- **청킹**: 800~1200 / overlap 100~200으로 시작, k=4~6 검증.
- **보안**: .env 키 관리 및 내부망에서 API만 공개.
- **관측**: LangSmith 추적, 필요한 경우 W&B 로깅 병행.
