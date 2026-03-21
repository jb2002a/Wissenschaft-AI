
"""
LLM judge 기반 metric 레거시 처리.


"""

# import dspy
#
# from src.translation.signatures.translation_judge import TranslationQualityJudge
#
# _judge = dspy.Predict(TranslationQualityJudge)
#
#
# def _to_1to5_int(value, default: int = 1) -> int:
#     """Convert model output to int score and clamp to [1, 5]."""
#     try:
#         parsed = int(round(float(value)))
#     except (TypeError, ValueError):
#         parsed = default
#     return max(1, min(5, parsed))
#
#
# def metric_llm(example, pred, trace=None) -> float:
#     """
#     MIPRO 등 옵티마이저용 metric. Judge 5개 항목의 평균(반올림)을 1~5 스칼라로 반환.
#
#     - example.original_text: 원문
#     - example.translated_text: 인간 번역문(정답)
#     - pred.translated_text: 모델 번역문
#     """
#     out = _judge(
#         source_text=example.original_text,
#         reference_text=example.translated_text,
#         candidate_text=pred.translated_text,
#     )
#
#     faithfulness = _to_1to5_int(getattr(out, "faithfulness", None))
#     terminology_accuracy = _to_1to5_int(getattr(out, "terminology_accuracy", None))
#     korean_fluency = _to_1to5_int(getattr(out, "korean_fluency", None))
#     style_register = _to_1to5_int(getattr(out, "style_register", None))
#     overall_score = _to_1to5_int(getattr(out, "overall_score", None))
#
#     parts = (
#         faithfulness,
#         terminology_accuracy,
#         korean_fluency,
#         style_register,
#         overall_score,
#     )
#     score = round(sum(parts) / 5)
#     return float(max(1, min(5, score)))
