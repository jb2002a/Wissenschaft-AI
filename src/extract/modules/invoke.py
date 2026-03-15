"""Sentence alignment invoke module using DSPy Predict."""

# This is temporary module for sentence alignment, we will replace it with a more efficient module in the future.

import json
from pathlib import Path

import dspy

from src.extract.signatures.sentence_align import ChunkAlignment

def _ensure_list(data: list[dict[str, str]] | str) -> list[dict[str, str]]:
    """Predict 결과가 문자열이면 JSON 파싱 후 리스트로 반환."""
    if isinstance(data, list):
        return data
    return json.loads(data)


def invoke(
    original_text: str,
    translated_text: str,
    output_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """의미적 청크 단위 1:1 매핑 예측을 수행하고, output_path가 주어지면 해당 경로에 JSON으로 저장한다."""
    if dspy.settings.lm is None:
        raise ValueError("DSPy LM이 설정되지 않았습니다.")

    predictor = dspy.Predict(ChunkAlignment)
    out = predictor(
        original_text=original_text,
        translated_text=translated_text,
    )
    pairs = _ensure_list(out.aligned_pairs)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

    return pairs


