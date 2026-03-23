from typing import List
from pathlib import Path
import json

import dspy

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TEST_JSON = _REPO_ROOT / "resources" / "test_dataset" / "test.json"


def load_test_dataset_json(path: Path = _DEFAULT_TEST_JSON) -> List[dict]:
    """테스트 데이터셋 파일을 로드해 List[dict] 형식으로 리턴"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def change_to_the_examples(items: List[dict]) -> List[dspy.Example]:
    """List[dict] 형식을 dspy.Example 형식으로 변환"""
    examples: List[dspy.Example] = []

    for item in items:
        example = dspy.Example(
            original_text=item["german"],
            translated_text=item["korean"],
        ).with_inputs("original_text")
        examples.append(example)
    return examples

