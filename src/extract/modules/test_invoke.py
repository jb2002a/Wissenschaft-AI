"""invoke 모듈 실제 테스트: resources의 md 파일을 텍스트화하여 의미적 청크 정렬 후 JSON 저장."""

# This will be removed in the future.

from pathlib import Path

from src.extract.modules.invoke import invoke
from src.extract.utils.md_to_text import md_to_text
from src.translation.modules.translate import get_lm


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    resources = root / "resources"

    ocr_path = resources / "ocr_result-18.36.md"
    translated_path = resources / "translated-18.36.md"
    output_path = resources / "aligned-18.36.json"

    original_text = md_to_text(ocr_path)
    translated_text = md_to_text(translated_path)

    get_lm()
    pairs = invoke(
        original_text=original_text,
        translated_text=translated_text,
        output_path=output_path,
    )
    print(f"매핑된 청크 쌍 수: {len(pairs)}")
    print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    main()
