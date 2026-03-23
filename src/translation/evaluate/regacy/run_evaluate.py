"""LLM-as-a-judge evaluation, 현재는 사용하지 않음.

실행 예 (프로젝트 루트에서):
  # 최적화 모듈 로드(기본)
  python -c "from src.translation.evaluate.run_evaluate import run_translation_evaluate; print(run_translation_evaluate())"
  # 최적화 모듈 로드 없이 기본 모듈 사용
  python -c "from src.translation.evaluate.run_evaluate import run_translation_evaluate; print(run_translation_evaluate(load=False))"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import dspy
from dspy.evaluate import Evaluate

from src.translation.metrics.translate_metric import metric_llm
from src.translation.modules.translate import TranslateModule, get_lm

_DEFAULT_TEST_JSON = (
    Path(__file__).resolve().parents[3] / "resources" / "test_dataset" / "test.json"
)
_DEFAULT_OPTIMIZED_ARTIFACT = (
    Path(__file__).resolve().parents[3] / "artifacts" / "translation_optimized.json"
)
_EVALUATION_LOGS_DIR = Path(__file__).resolve().parents[3] / "logs" / "evaluation_logs"


def load_test_devset(json_path: str | Path | None = None) -> List[dspy.Example]:
    """
    test.json을 로드해 dspy.Example 리스트로 변환한다.
    JSON 각 항목: {"german": str, "korean": str} -> original_text, translated_text.
    """
    path = Path(json_path) if json_path is not None else _DEFAULT_TEST_JSON
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else [data]
    return [
        dspy.Example(
            original_text=r["german"],
            translated_text=r["korean"],
        ).with_inputs("original_text")
        for r in rows
    ]


def run_translation_evaluate(
    dataset_path: str | Path | None = None,
    num_threads: int | None = 1,
    display_progress: bool = False,
    display_table: bool | int = False,
    load: bool = True,
    optimized_path: str | Path | None = None,
    lm_type: str = "gemini",
) -> List[dict]:
    """
    DSPy Evaluate로 테스트 데이터셋에 대해 번역 품질 평가를 수행하고,
    샘플별 original_text, translated_text, ai_text 및 5개 점수를 담은 dict 리스트를 반환·저장한다.

    - devset: dataset_path(기본 resources/test_dataset/test.json)에서 로드한 Example 목록
    - program:
        - load=True  이면 artifacts/translation_optimized.json(또는 optimized_path)을 load한 TranslateModule
        - load=False 이면 기본 TranslateModule 인스턴스
    - metric: metric_llm(..., evaluation=True)로 5개 항목 dict 반환, Evaluate 내부 합산용으로는 평균값 전달

    Returns:
        list[dict]: 각 샘플별 original_text, translated_text, ai_text, faithfulness, terminology_accuracy, korean_fluency, style_register, overall_score
    """
    get_lm(lm_type)
    devset = load_test_devset(dataset_path)
    collector: List[dict] = []

    def _wrapper_metric(example: dspy.Example, pred: dspy.Prediction, trace=None):
        scores = metric_llm(example, pred, trace=trace, evaluation=True)
        row = {
            "original_text": example.original_text,
            "translated_text": example.translated_text,
            "ai_text": getattr(pred, "translated_text", ""),
            **scores,
        }
        collector.append(row)
        return sum(scores.values()) / 5.0

    program = TranslateModule()
    if load:
        artifact_path = Path(optimized_path) if optimized_path is not None else _DEFAULT_OPTIMIZED_ARTIFACT
        program.load(str(artifact_path))
    evaluator = Evaluate(
        devset=devset,
        metric=_wrapper_metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=display_table,
    )
    evaluator(program)

    _EVALUATION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _EVALUATION_LOGS_DIR / f"log_data_{timestamp}.txt"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(collector, f, ensure_ascii=False, indent=2)

    return collector
