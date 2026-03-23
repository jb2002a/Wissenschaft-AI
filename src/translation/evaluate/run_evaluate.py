"""번역 평가 실행 judge model : claude"""

import dspy
import wandb

from src.translation.modules.translate_dataset import load_test_dataset_json, change_to_the_examples
from src.translation.metrics.translate_metric_llm import metric_llm_v2
from src.translation.modules.translate import TranslateModule
from src.translation.modules.get_lm import get_lm


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_row(row):
    """DSPy EvaluationResult.results row를 버전별로 안전 파싱."""
    source = None
    reference = None
    prediction = None
    score = None

    # 케이스 1) tuple/list: (example, prediction, score) 형태
    if isinstance(row, (tuple, list)):
        ex = row[0] if len(row) > 0 else None
        pred = row[1] if len(row) > 1 else None
        score = row[2] if len(row) > 2 else None

        source = _safe_get(ex, "original_text")
        reference = _safe_get(ex, "translated_text")
        prediction = _safe_get(pred, "translated_text")

        return source, reference, prediction, score

    # 케이스 2) dict 형태
    if isinstance(row, dict):
        ex = row.get("example")
        pred = row.get("prediction")
        score = row.get("score", row.get("metric"))

        source = (
            _safe_get(ex, "original_text")
            or row.get("source")
            or row.get("original_text")
            or row.get("input")
        )
        reference = (
            _safe_get(ex, "translated_text")
            or row.get("reference")
            or row.get("translated_text")
            or row.get("label")
        )
        prediction = (
            _safe_get(pred, "translated_text")
            or row.get("prediction")
            or row.get("prediction_text")
            or row.get("output")
        )

        return source, reference, prediction, score

    # 케이스 3) 객체 형태
    ex = _safe_get(row, "example")
    pred = _safe_get(row, "prediction")
    score = _safe_get(row, "score", _safe_get(row, "metric"))

    source = _safe_get(ex, "original_text", _safe_get(row, "original_text"))
    reference = _safe_get(ex, "translated_text", _safe_get(row, "translated_text"))
    prediction = _safe_get(pred, "translated_text", _safe_get(row, "prediction_text"))

    # 마지막 fallback
    if prediction is None:
        prediction = str(row)

    return source, reference, prediction, score


def run_evaluate(lm_type: str = "claude") -> dspy.evaluate.EvaluationResult:
    get_lm(lm_type)
    items = load_test_dataset_json()
    examples = change_to_the_examples(items)

    run = wandb.init(
        project="wissenschaft-translation-eval",
        # API 키가 유효하면 online으로 동작
        config={"lm_type": lm_type, "dataset_size": len(examples), "metric": "metric_llm_v2"},
    )

    try:
        evaluator = dspy.Evaluate(devset=examples, metric=metric_llm_v2)
        result = evaluator(program=TranslateModule())

        # 1) 요약 메트릭
        payload = {
            "eval/score": _safe_get(result, "score"),
            "eval/num_examples": len(examples),
        }
        wandb.log({k: v for k, v in payload.items() if v is not None})

        # 2) 문장별 테이블
        rows = _safe_get(result, "results", [])
        table = wandb.Table(columns=["idx", "source", "reference", "prediction", "score"])

        for idx, row in enumerate(rows):
            source, reference, prediction, score = _extract_row(row)
            table.add_data(idx, source, reference, prediction, score)

        wandb.log({"eval/per_example_table": table})

        # 디버깅용: 결과 구조 확인
        wandb.summary["results_count"] = len(rows)

        return result
    finally:
        wandb.finish()


if __name__ == "__main__":
    result = run_evaluate()
    print(result)