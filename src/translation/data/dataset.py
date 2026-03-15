# src/translation/data/dataset.py
# load dataset from json file and convert to dspy.Example objects

import json
from pathlib import Path

import dspy

_json_path = Path(__file__).resolve().parents[3] / "resources" / "initial_test_json_file.json"
with _json_path.open(encoding="utf-8") as f:
    _data = json.load(f)

# 단일 객체면 리스트로 감싼다.
rows = _data if isinstance(_data, list) else [_data]

#List of dspy.Example objects
trainset = [
    dspy.Example(
        original_text=r["original_text"],
        translated_text=r["translated_text"],
    ).with_inputs("original_text")
    for r in rows
]