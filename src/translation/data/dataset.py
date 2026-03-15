# load dataset from json file and convert to dspy.Example objects

import json
import random
from pathlib import Path
from typing import List, Tuple

import dspy

_JSON_PATH = (
    Path(__file__).resolve().parents[3] / "resources" / "mapping_dataset" / "merged_mapping.json"
)


def _load_examples() -> List[dspy.Example]:
    """Load merged_mapping.json and convert to dspy.Example list. JSON keys: original, translation."""
    with _JSON_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else [data]
    return [
        dspy.Example(
            original_text=r["original"],
            translated_text=r["translation"],
        ).with_inputs("original_text")
        for r in rows
    ]


def get_train_valset(
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Return (trainset, valset). Size is determined by dataset length and train_ratio (e.g. 0.8 = 4:1 train:val).
    """
    examples = _load_examples()
    n = len(examples)
    if n == 0:
        raise ValueError("Dataset is empty.")
    train_size = int(n * train_ratio)
    val_size = n - train_size
    if val_size <= 0:
        raise ValueError(f"train_ratio={train_ratio} leaves no validation set (n={n}).")
    if shuffle:
        rng = random.Random(seed)
        shuffled = list(examples)
        rng.shuffle(shuffled)
        trainset = shuffled[:train_size]
        valset = shuffled[train_size:]
    else:
        trainset = examples[:train_size]
        valset = examples[train_size:]
    return trainset, valset