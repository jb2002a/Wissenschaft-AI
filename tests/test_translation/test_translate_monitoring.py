"""translate 모듈 단일 문장 번역 (optimized 모듈 로드 후 실행)."""

# 실행 CLI (프로젝트 루트에서):
# python -m tests.test_translation.test_translate_monitoring

import os
import uuid

from dotenv import load_dotenv

load_dotenv()

_OPTIMIZED_ARTIFACT = "artifacts/translation_optimized.json"


def run_translate_single_sentence() -> str | None:
    """translation_optimized.json을 로드한 뒤 해당 모듈로 한 문장을 번역한다. 조건 미충족 시 None 반환."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY가 없어 번역을 건너뜁니다.")
        return None
    if not os.path.isfile(_OPTIMIZED_ARTIFACT):
        print(f"{_OPTIMIZED_ARTIFACT}이 없어 최적화된 모듈 번역을 건너뜁니다.")
        return None

    try:
        from src.translation.modules.translate import translate_with_optimized
    except ModuleNotFoundError as exc:
        print(f"필수 모듈이 없어 건너뜁니다: {exc}")
        return None

    source_sentence = f"Das Wetter ist heute sehr gut. run-{uuid.uuid4().hex[:8]}"
    translated_text = translate_with_optimized(
        source_sentence, optimized_path=_OPTIMIZED_ARTIFACT
    )
    return translated_text


if __name__ == "__main__":
    result = run_translate_single_sentence()
    if result is not None:
        print(result)
