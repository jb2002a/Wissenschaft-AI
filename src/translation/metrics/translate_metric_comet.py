"""
COMET 기반 번역 품질 metric (wmt22-comet-da, gpus=0 고정).

DSPy MIPROv2 컴파일 시 `metric(example, pred, trace=None) -> float` 형태로 사용된다.
"""

from __future__ import annotations

from typing import Any

from comet import download_model, load_from_checkpoint


_COMET_MODEL = None


def _get_comet_model():
    global _COMET_MODEL
    if _COMET_MODEL is None:
        model_path = download_model("wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(model_path)
    return _COMET_MODEL


def metric_comet(example: Any, pred: Any, trace: Any = None) -> float:
    """
    MIPROv2용 metric.

    - example.original_text: 독일어 원문
    - example.translated_text: 정답(참조) 한국어
    - pred.translated_text: 후보 한국어
    """

    sample = [
        {
            "src": example.original_text,
            "mt": pred.translated_text,
            "ref": example.translated_text,
        }
    ]

    model = _get_comet_model()
    model_output = model.predict(sample, batch_size=1, gpus=0)

    return float(model_output.scores[0])

