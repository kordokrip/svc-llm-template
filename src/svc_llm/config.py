from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str

    # --- Embedding ---
    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-large"

    # --- Pinecone (optional) ---
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index: str | None = None

    # --- LangSmith / LangChain tracing ---
    langsmith_tracing: bool = True
    langsmith_api_key: str | None = None
    langsmith_project: str = "SVC-LLM"
    # .env 에 자주 쓰는 LANGCHAIN_TRACING_V2도 받아둡니다(불리언 파싱됨)
    langchain_tracing_v2: bool = True

    # --- Weights & Biases (optional) ---
    wandb_api_key: str | None = None
    wandb_project: str | None = None

    # .env 읽기 + 정의되지 않은 키는 무시
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()