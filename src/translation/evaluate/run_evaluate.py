"""번역 평가 실행 judge model : claude"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import dspy

from src.translation.modules.translate_dataset import load_test_dataset_json, change_to_the_examples
from src.translation.metrics.translate_metric_llm import metric_llm_v2
from src.translation.modules.translate import TranslateModule
from src.translation.modules.get_lm import get_lm


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOG_DIR = _REPO_ROOT / "logs" / "evaluation_json"


def _write_json_log(
    result: dspy.evaluate.EvaluationResult,
    lm_type: str,
    dataset_size: int,
    log_dir: Path = _DEFAULT_LOG_DIR,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)

    # result.results: list[tuple[dspy.Example, dspy.Example, Any]]
    per_example: list[dict] = []
    for idx, (example, pred, score) in enumerate(result.results):
        per_example.append(
            {
                "idx": idx,
                "source": example.original_text,
                "reference": example.translated_text,
                "prediction": pred.translated_text,
                "score": float(score),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = log_dir / f"evaluation_{lm_type}_{timestamp}.json"

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "lm_type": lm_type,
            "metric": "metric_llm_v2",
            "dataset_size": dataset_size,
            "num_results": len(per_example),
        },
        "summary": {
            "score": float(result.score),
        },
        "per_example": per_example,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


def run_evaluate(lm_type: str = "claude") -> dspy.evaluate.EvaluationResult:
    get_lm(lm_type)
    items = load_test_dataset_json()
    examples = change_to_the_examples(items)

    evaluator = dspy.Evaluate(devset=examples, metric=metric_llm_v2)
    result = evaluator(program=TranslateModule())

    json_path = _write_json_log(result=result, lm_type=lm_type, dataset_size=len(examples))
    print(f"[evaluation-json] saved: {json_path}")

    return result


if __name__ == "__main__":
    result = run_evaluate()
    print(result)