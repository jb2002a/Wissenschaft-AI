"""translate 모듈 단일 문장 번역 + Langtrace 모니터링 통합 테스트."""

import os
import unittest
import uuid

from dotenv import load_dotenv

load_dotenv()

try:
    from langtrace_python_sdk import langtrace, with_langtrace_root_span
except ModuleNotFoundError:
    langtrace = None
    with_langtrace_root_span = None

_LANGTRACE_INITIALIZED = False


def _init_langtrace_once() -> None:
    """테스트 프로세스에서 Langtrace를 1회만 초기화한다."""
    global _LANGTRACE_INITIALIZED

    if _LANGTRACE_INITIALIZED:
        return

    if langtrace is None:
        raise unittest.SkipTest("langtrace_python_sdk가 설치되지 않아 테스트를 건너뜁니다.")

    langtrace_api_key = os.getenv("LANGTRACE_API_KEY")
    if not langtrace_api_key:
        raise unittest.SkipTest("LANGTRACE_API_KEY가 없어 모니터링 테스트를 건너뜁니다.")

    langtrace.init(
        api_key=langtrace_api_key,
        service_name="wissenschaft-ai-translation-tests",
    )
    _LANGTRACE_INITIALIZED = True


class TestTranslateMonitoring(unittest.TestCase):
    """translate 모듈의 단일 번역 호출이 추적되도록 보장한다."""

    def test_translate_single_sentence_with_langtrace_monitoring(self) -> None:
        """한 문장을 번역하고 Langtrace에서 추적 가능한 호출을 생성한다."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise unittest.SkipTest("GOOGLE_API_KEY가 없어 번역 테스트를 건너뜁니다.")

        try:
            from src.translation.modules.translate import translate
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"필수 모듈이 없어 테스트를 건너뜁니다: {exc}") from exc

        _init_langtrace_once()

        source_sentence = f"Das Wetter ist heute sehr gut. run-{uuid.uuid4().hex[:8]}"

        if with_langtrace_root_span is None:
            raise unittest.SkipTest("langtrace root span 데코레이터를 사용할 수 없습니다.")

        @with_langtrace_root_span(name="test_translate_single_sentence")
        def _run_translation() -> str:
            return translate(source_sentence)

        translated_text = _run_translation()

        self.assertIsInstance(translated_text, str)
        self.assertNotEqual(translated_text.strip(), "")
