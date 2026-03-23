"""
LLM judge 기반 metric. MIPRO 등 옵티마이저용 1~5 스칼라 점수.
"""

import dspy

from src.translation.signatures.translation_judge import TranslationQualityJudge

_judge = dspy.Predict(TranslationQualityJudge)


def metric_llm_v2(example, pred, trace=None) -> float:
    out = _judge(
        source_text=example.original_text,
        reference_text=example.translated_text,
        candidate_text=pred.translated_text,
    )

    # 1. 정수로 반올림하지 말고 raw float 값을 사용 (0~1 사이로 정규화 권장)
    def to_float(value):
        try: return float(value)
        except: return 1.0

    f = to_float(getattr(out, "faithfulness", 1.0))
    t = to_float(getattr(out, "terminology_accuracy", 1.0))
    fl = to_float(getattr(out, "korean_fluency", 1.0))
    s = to_float(getattr(out, "style_register", 1.0))
    o = to_float(getattr(out, "overall_score", 1.0))

    # 2. 가중치 부여 (학술 번역은 정확성과 용어가 훨씬 중요함)
    # 충실성(40%) + 용어(30%) + 유창성(10%) + 스타일(10%) + 종합(10%)
    weighted_score = (f * 0.4) + (t * 0.3) + (fl * 0.1) + (s * 0.1) + (o * 0.1)
    
    # 3. 0~100점 스케일로 변환 (Optimizer가 변화를 더 잘 감지함)
    final_score = (weighted_score / 5.0) * 100.0
    
    return final_score
