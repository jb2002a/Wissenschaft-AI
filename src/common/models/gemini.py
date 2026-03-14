"""Gemini 3.1 Pro 기반 LangChain Chat 모델."""

import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 3.1 Pro 공식 모델 ID (Google AI for Developers 기준)
GEMINI_31_PRO_MODEL_ID = "gemini-3.1-pro-preview"


def get_gemini_31_pro_llm(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Gemini 3.1 Pro를 사용하는 LangChain Chat 모델 인스턴스를 반환합니다.

    환경 변수 GOOGLE_API_KEY 또는 GEMINI_API_KEY가 있으면 자동으로 사용합니다.
    api_key를 인자로 넘기면 그 값을 우선 사용합니다.

    Args:
        model: 모델 ID. None이면 GEMINI_31_PRO_MODEL_ID 사용.
        api_key: Google AI API 키. None이면 환경 변수 사용.
        temperature: 생성 온도 (0.0~2.0).
        **kwargs: ChatGoogleGenerativeAI에 그대로 전달 (예: max_output_tokens).

    Returns:
        ChatGoogleGenerativeAI 인스턴스.
    """
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model or GEMINI_31_PRO_MODEL_ID,
        api_key=resolved_key,
        temperature=temperature,
        **kwargs,
    )
