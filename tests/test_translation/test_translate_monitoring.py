"""translate 모듈 전체 텍스트 번역 모니터링 스크립트."""

# 실행 CLI (프로젝트 루트에서):
# python -m tests.test_translation.test_translate_monitoring

import os

from dotenv import load_dotenv

load_dotenv()

_OPTIMIZED_ARTIFACT = "artifacts/translation_optimized.json"
_SOURCE_MD = "tests/test_translation/original_text_data.md"


def run_translate_single_sentence() -> str | None:
    """optimized 모듈로 markdown 원문 전체를 한 번에 번역해 문자열로 반환한다."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY가 없어 번역을 건너뜁니다.")
        return None
    if not os.path.isfile(_OPTIMIZED_ARTIFACT):
        print(f"{_OPTIMIZED_ARTIFACT}이 없어 최적화된 모듈 번역을 건너뜁니다.")
        return None
    if not os.path.isfile(_SOURCE_MD):
        print(f"{_SOURCE_MD}이 없어 원문 번역을 건너뜁니다.")
        return None

    try:
        from src.translation.modules.translate import translate_with_optimized
    except ModuleNotFoundError as exc:
        print(f"필수 모듈이 없어 건너뜁니다: {exc}")
        return None

    try:
        with open(_SOURCE_MD, "r", encoding="utf-8") as f:
            source_text = f.read().strip()
    except OSError as exc:
        print(f"원문 파일 로드 실패로 번역을 건너뜁니다: {exc}")
        return None

    if not source_text:
        print("원문 파일이 비어 있어 번역을 건너뜁니다.")
        return None

    return translate_with_optimized(source_text, optimized_path=_OPTIMIZED_ARTIFACT)


if __name__ == "__main__":
    result = run_translate_single_sentence()
    if result is not None:
        print(result)
