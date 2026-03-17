
import dspy

from src.translation.signatures.translation_judge import TranslationQualityJudge


_judge = dspy.Predict(TranslationQualityJudge)


def _to_1to5_int(value, default: int = 1) -> int:
    """Convert model output to int score and clamp to [1, 5]."""
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(5, parsed))


def _build_evaluation_scores(
    faithfulness: int,
    terminology_accuracy: int,
    korean_fluency: int,
    style_register: int,
    overall_score: int,
) -> dict[str, int]:
    return {
        "faithfulness": faithfulness,
        "terminology_accuracy": terminology_accuracy,
        "korean_fluency": korean_fluency,
        "style_register": style_register,
        "overall_score": overall_score,
    }


def metric_llm(example, pred, trace=None, evaluation: bool = False) -> float | dict[str, int]:
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

    faithfulness = _to_1to5_int(getattr(out, "faithfulness", None))
    terminology_accuracy = _to_1to5_int(getattr(out, "terminology_accuracy", None))
    korean_fluency = _to_1to5_int(getattr(out, "korean_fluency", None))
    style_register = _to_1to5_int(getattr(out, "style_register", None))
    overall_score = _to_1to5_int(getattr(out, "overall_score", None))

    evaluation_scores = _build_evaluation_scores(
        faithfulness=faithfulness,
        terminology_accuracy=terminology_accuracy,
        korean_fluency=korean_fluency,
        style_register=style_register,
        overall_score=overall_score,
    )

    if evaluation:
        return evaluation_scores

    # 소수점 없이 1~5 정수 점수만 사용하기 위해 평균 후 반올림
    score = round(sum(evaluation_scores.values()) / 5)

    return float(max(1, min(5, score)))