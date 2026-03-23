import os

import dspy
from dotenv import load_dotenv

load_dotenv()


def get_lm_gemini() -> None:
    """DSPy LM을 로드하고 전역으로 설정한다. (Gemini)"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API 키가 설정되지 않았습니다. (GOOGLE_API_KEY)")

    lm = dspy.LM(
        "gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)


def get_lm_claude() -> None:
    """DSPy LM을 로드하고 전역으로 설정한다. (Claude)"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API 키가 설정되지 않았습니다. (ANTHROPIC_API_KEY)")

    lm = dspy.LM(
        "anthropic/claude-sonnet-4-5-20250929",
        api_key=api_key,
        temperature=0.0,
    )
    dspy.configure(lm=lm)


def get_lm(lm_type: str = "gemini") -> None:
    """lm_type에 따라 DSPy LM을 로드하고 전역으로 설정한다."""
    normalized = (lm_type or "gemini").strip().lower()
    if normalized == "gemini":
        get_lm_gemini()
    elif normalized == "claude":
        get_lm_claude()
    else:
        raise ValueError("유효하지 않은 LM 타입입니다. (gemini/claude)")

