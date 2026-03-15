# src/translation/metrics/translate_metric.py
import dspy

from src.translation.signatures.translation_judge import TranslationQualityJudge


_judge = dspy.Predict(TranslationQualityJudge)


def metric_llm(example, pred, trace=None) -> float:
    """
    Optimizer용 metric 시그니처:
    - example.original_text: 원문
    - example.translated_text: 인간 번역문(정답)
    - pred.translated_text: 모델 번역문
    """
    out = _judge(
        source_text=example.original_text,
        reference_text=example.translated_text,
        candidate_text=pred.translated_text,
    )

    # 모델 응답이 문자열로 오더라도 안전하게 float 변환
    try:
        score = float(out.score)
    except (TypeError, ValueError):
        score = 0.0

    return max(0.0, min(1.0, score))