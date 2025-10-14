from __future__ import annotations
import os
import argparse
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# 1) Load .env if present (does NOT override real env vars)
load_dotenv(find_dotenv(), override=False)

# 2) Defaults (safe fallbacks)
DEFAULT_FT_MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:svc-41mini-sft-dpo-80usd-sft:CLB4qudK"
DEFAULT_PROMPT = "모델 확인용: 'SVC 절차 3줄 요약 + 마지막 한 줄 명령' 형식으로 응답하세요."


def parse_args():
    ap = argparse.ArgumentParser(description="FT 모델 스모크 테스트")
    ap.add_argument("--model", default=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_FT_MODEL),
                    help="사용할 모델 ID (미설정 시 .env 또는 기본 FT 모델 사용)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="테스트 프롬프트")
    ap.add_argument("--temp", type=float, default=0.2, help="temperature")
    ap.add_argument("--stream", action="store_true", help="스트리밍 모드 사용")
    return ap.parse_args()


def require_env(key: str):
    val = os.getenv(key)
    if not val:
        here = Path.cwd()
        env_path = find_dotenv() or ".env(없음)"
        raise RuntimeError(
            f"환경변수 {key} 가 설정되지 않았습니다.\n"
            f"- 현재 경로: {here}\n"
            f"- 로드된 .env: {env_path}\n"
            f"해결: 프로젝트 루트에 .env 파일을 만들고 아래와 같이 설정 후 다시 실행하세요.\n"
            f"OPENAI_API_KEY=sk-...\nOPENAI_CHAT_MODEL={DEFAULT_FT_MODEL}\n"
        )
    return val


def main():
    args = parse_args()

    # 3) 최소 요구: API 키 검증
    require_env("OPENAI_API_KEY")

    # 4) 모델 결정 (CLI > ENV > 기본값)
    model_id = args.model or DEFAULT_FT_MODEL

    print("=== FT Model Smoke Test ===")
    print("Model:", model_id)

    chat = ChatOpenAI(model=model_id, temperature=args.temp, streaming=args.stream)

    if args.stream:
        print("Streaming reply:")
        text = ""
        for chunk in chat.stream(args.prompt):
            piece = chunk.content or ""
            text += piece
            print(piece, end="", flush=True)
        print("\n---\nFULL:\n", text)
    else:
        resp = chat.invoke(args.prompt)
        print("Reply:\n", resp.content)


if __name__ == "__main__":
    main()