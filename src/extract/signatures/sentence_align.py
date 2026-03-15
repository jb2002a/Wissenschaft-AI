"""원문·번역본 의미적 청크 단위 1:1 매핑용 DSPy 시그니처."""

import dspy


class ChunkAlignment(dspy.Signature):
    """원문 텍스트와 번역본 텍스트를 의미적인 청크 단위로 1:1 매핑한다. 각 원문 청크에 대응하는 번역 청크를 짝지어 JSON 리스트로 반환한다."""

    original_text: str = dspy.InputField(desc="원문 텍스트(전체)")
    translated_text: str = dspy.InputField(desc="번역본 텍스트(전체)")

    aligned_pairs: list[dict[str, str]] = dspy.OutputField(
        desc="의미적 청크 단위 1:1 매핑 결과. JSON 배열 형태로 반환. 예: [{\"original\": \"원문 청크1\", \"translated\": \"번역 청크1\"}, ...]. 반드시 유효한 JSON list만 출력한다."
    )
