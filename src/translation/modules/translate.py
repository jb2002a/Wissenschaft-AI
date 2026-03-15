"""LM 로드(get_lm), predict(translate), 및 MIPROv2용 dspy.Module(student)."""

import os

import dspy
from dotenv import load_dotenv

from src.translation.signatures.german_to_korean import GermanToKorean

load_dotenv()

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"


class TranslateModule(dspy.Module):
    """독일어→한국어 번역 DSPy 모듈. MIPROv2.compile(student=...)에 전달용."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(GermanToKorean)

    def forward(self, original_text: str):
        out = self.predictor(original_text=original_text)
        return out


def get_lm() -> None:
    """DSPy LM을 로드하고 전역으로 설정한다. 기본: anthropic/claude-sonnet-4-5."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
    lm = dspy.LM(_DEFAULT_MODEL, api_key=api_key, temperature=0.0)
    dspy.configure(lm=lm)


# translate와 show_last_promp는 optimizer 로직에서 사용하지 않음.

def translate(original_text: str) -> str:
    """프롬프트를 LM에 보내고 predict한 응답 문자열을 반환한다."""
    
    get_lm()

    predictor = dspy.Predict(GermanToKorean)

    out = predictor(original_text=original_text)
    return out.translated_text

def show_last_prompt(n: int = 1) -> None:
    """마지막 n회 호출의 자동 생성 프롬프트와 응답을 콘솔에 출력한다.
    translate() 호출 후에 호출하면 된다."""
    if dspy.settings.lm is None:
        raise RuntimeError("LM이 아직 로드되지 않았습니다. get_lm() 후 translate()를 먼저 호출하세요.")
    dspy.settings.lm.inspect_history(n=n)

