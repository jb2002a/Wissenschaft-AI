"""번역 모듈, model : gemini"""

import dspy

from src.translation.modules.get_lm import get_lm
from src.translation.signatures.german_to_korean import GermanToKorean
import os


class TranslateModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(GermanToKorean)
        self.translate_lm = dspy.LM(
            "gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0,
        )

    def forward(self, original_text: str):
        with dspy.context(lm=self.translate_lm):
            return self.predictor(original_text=original_text)

# 하단 코드는 사용하지 않는 코드입니다. 필요시 사용

def translate(original_text: str, lm_type: str = "gemini") -> str:
    """번역 실행 모듈, 기본 모델로 번역"""
    
    get_lm(lm_type)
    predictor = dspy.Predict(GermanToKorean)

    out = predictor(original_text=original_text)
    return out.translated_text


def translate_with_optimized(
    original_text: str,
    optimized_path: str = "artifacts/translation_optimized.json",
    lm_type: str = "gemini",
) -> str:
    """optimizer 모듈 로드 후 번역 진행"""
    get_lm(lm_type)
    module = TranslateModule()
    module.load(optimized_path)
    out = module(original_text=original_text)
    return out.translated_text

def show_last_prompt(n: int = 1) -> None:
    """마지막 n회 호출의 자동 생성 프롬프트와 응답을 콘솔에 출력한다.
    translate() 호출 후에 호출하면 된다."""
    if dspy.settings.lm is None:
        raise RuntimeError("LM이 아직 로드되지 않았습니다. get_lm() 후 translate()를 먼저 호출하세요.")
    dspy.settings.lm.inspect_history(n=n)

