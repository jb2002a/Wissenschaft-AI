"""독일어→한국어 번역용 DSPy 시그니처."""

import dspy


class GermanToKorean(dspy.Signature):
    """Translate German philosophical academic books into natural Korean. Output must be written entirely in Korean."""

    original_text: str = dspy.InputField(desc="German text to translate")
    translated_text: str = dspy.OutputField(desc="Korean translation of the input German text")
