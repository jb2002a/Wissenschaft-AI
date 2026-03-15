
import dspy

from src.translation.signatures.translation_judge import TranslationQualityJudge


_judge = dspy.Predict(TranslationQualityJudge)
_metric_cache: dict[tuple[str, str, str], float] = {}


def _to_1to5_int(value, default: int = 1) -> int:
    """Convert model output to int score and clamp to [1, 5]."""
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(5, parsed))


def metric_llm(example, pred, trace=None) -> float:
    """
    Optimizer용 metric 시그니처:
    - example.original_text: 원문
    - example.translated_text: 인간 번역문(정답)
    - pred.translated_text: 모델 번역문
    """
    source_text = str(getattr(example, "original_text", ""))
    reference_text = str(getattr(example, "translated_text", ""))
    candidate_text = str(getattr(pred, "translated_text", ""))
    cache_key = (source_text, reference_text, candidate_text)

    if cache_key in _metric_cache:
        return _metric_cache[cache_key]

    out = _judge(
        source_text=source_text,
        reference_text=reference_text,
        candidate_text=candidate_text,
    )

    faithfulness = _to_1to5_int(getattr(out, "faithfulness", None))
    terminology_accuracy = _to_1to5_int(getattr(out, "terminology_accuracy", None))
    korean_fluency = _to_1to5_int(getattr(out, "korean_fluency", None))
    style_register = _to_1to5_int(getattr(out, "style_register", None))
    overall_score = _to_1to5_int(getattr(out, "overall_score", None))

    # 소수점 없이 1~5 정수 점수만 사용하기 위해 평균 후 반올림
    score = round(
        (
            faithfulness
            + terminology_accuracy
            + korean_fluency
            + style_register
            + overall_score
        )
        / 5
    )

    final_score = float(max(1, min(5, score)))
    _metric_cache[cache_key] = final_score
    return final_score