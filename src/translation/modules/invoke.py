"""LM 로드(get_lm) 및 predict(invoke)."""

import os

import dspy
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_MODEL = "gemini/gemini-2.0-flash"

def get_lm() -> None:
    """DSPy LM을 로드하고 전역으로 설정한다. 기본: gemini/gemini-2.0-flash."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
    lm = dspy.LM(_DEFAULT_MODEL, api_key=api_key, temperature=0.0)
    dspy.configure(lm=lm)


def invoke(input_text: str) -> str:
    """프롬프트를 LM에 보내고 predict한 응답 문자열을 반환한다."""
    if dspy.settings.lm is None:
        get_lm()

    predictor= dspy.Predict(
        dspy.Signature(
        "input_text -> response : str",
        )
    )

    out = predictor(input_text=input_text)
    return out.response


# __main__
if __name__ == "__main__":
    print(invoke("DSPy가 Langchain보다 강점을 가지고 있는 이유를 설명해줘"))
