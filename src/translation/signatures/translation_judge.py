import dspy


class TranslationQualityJudge(dspy.Signature):
    """Evaluate how well the candidate translation matches the human reference for the same German source. Goal: improve AI translation to be closer to the expert translator's output. Score by: (1) faithfulness to the source meaning, (2) terminology and factual accuracy, (3) style and readability in Korean, (4) overall similarity to the reference translation. Output exactly one number in [0.0, 1.0]; no other text or explanation."""
    source_text: str = dspy.InputField(desc="Original German source text")
    reference_text: str = dspy.InputField(desc="Expert human translator's Korean reference")
    candidate_text: str = dspy.InputField(desc="Model-generated Korean translation to score")
    score: float = dspy.OutputField(
        desc="Single number in [0.0, 1.0] only. No other text."
    )