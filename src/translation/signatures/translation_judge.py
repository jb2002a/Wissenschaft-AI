# src/translation/signatures/translation_judge.py
import dspy


class TranslationQualityJudge(dspy.Signature):
    """Score translation quality from 0.0 to 1.0 based on faithfulness, accuracy, and naturalness."""
    source_text: str = dspy.InputField(desc="Original German source text")
    reference_text: str = dspy.InputField(desc="Human translator's Korean reference")
    candidate_text: str = dspy.InputField(desc="Model-generated Korean translation")
    score: float = dspy.OutputField(
        desc="Single numeric score in [0.0, 1.0]. Return only the score."
    )