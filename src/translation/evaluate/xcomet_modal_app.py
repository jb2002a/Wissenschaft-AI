"""modal run -m src.translation.evaluate.xcomet_modal_app"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("wissenschaft-xcomet")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = "Unbabel/XCOMET-XL"
_LOG = logging.getLogger(__name__)

xcomet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unbabel-comet",
        "torch",
        "dspy",
    )
)


@app.function(
    image=xcomet_image,
    gpu="T4",
    timeout=60 * 60,
    memory=16384,
    secrets=[modal.Secret.from_name("huggingface")],
)
def score_xcomet_remote(
    samples: list[dict],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 4,
) -> dict:
    from comet import download_model, load_from_checkpoint

    _LOG.info("model download 시작")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    out = model.predict(samples, batch_size=batch_size, gpus=1)
    _LOG.info("predict 완료")
    return {
        "scores": [float(s) for s in out.scores],
        "system_score": float(out.system_score),
        "error_spans": out.metadata.error_spans,
    }


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


# 프로젝트 루트에서: modal run src/translation/evaluate/xcomet_modal_app.py


@app.local_entrypoint()
def main() -> None:
    configure_logging()
    _LOG.info("평가 파이프라인 시작")
    from src.translation.evaluate.test_json_eval import (
        report_to_plain_text,
        run_test_json_evaluation,
    )

    report = run_test_json_evaluation(optimized_path="artifacts/translation_optimized.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        _REPO_ROOT / "logs" / "evaluation_logs" / f"evaluation_report_{timestamp}.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_to_plain_text(report), encoding="utf-8")
    print(f"저장됨: {out_path}")
