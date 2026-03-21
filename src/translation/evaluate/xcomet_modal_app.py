"""Modal에서만 XCOMET 점수 계산 (GPU). `modal deploy` / `modal run` 진입점."""

from __future__ import annotations

import modal

app = modal.App("wissenschaft-xcomet")

_DEFAULT_MODEL = "Unbabel/XCOMET-XL"

xcomet_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unbabel-comet",
        "torch",
        "dspy"
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
    batch_size: int = 8,
) -> dict:
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    out = model.predict(samples, batch_size=batch_size, gpus=1)

    return {
        "scores": [float(s) for s in out.scores],
        "system_score": float(out.system_score),
        "error_spans": out.metadata.error_spans,
    }


# modal run -m src.translation.evaluate.xcomet_modal_app

@app.local_entrypoint()
def main():
    dummy = [{"src": "a", "mt": "b", "ref": "c"}]
    result = score_xcomet_remote.remote(dummy)
    print(result)