"""
LLM judge 시그니처(레거시) 주석 처리.

요청대로 TranslationQualityJudge 기반 시그니처를 사용하지 않도록
주석 처리해 둡니다.
"""

# import dspy
#
#
# class TranslationQualityJudge(dspy.Signature):
#     """Evaluate candidate translation quality with decomposed scores. Return only integer scores from 1 to 5 for each field, with no explanation."""
#     source_text: str = dspy.InputField(desc="Original German source text")
#     reference_text: str = dspy.InputField(desc="Expert human translator's Korean reference")
#     candidate_text: str = dspy.InputField(desc="Model-generated Korean translation to score")
#     faithfulness: int = dspy.OutputField(
#         desc="Meaning preservation from source. Integer in [1, 5]"
#     )
#     terminology_accuracy: int = dspy.OutputField(
#         desc="Terminology and factual precision. Integer in [1, 5]"
#     )
#     korean_fluency: int = dspy.OutputField(
#         desc="Natural Korean readability and coherence. Integer in [1, 5]"
#     )
#     style_register: int = dspy.OutputField(
#         desc="Academic style/register appropriateness. Integer in [1, 5]"
#     )
#     overall_score: int = dspy.OutputField(
#         desc="Overall quality considering all aspects. Integer in [1, 5]"
#     )