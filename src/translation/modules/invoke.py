"""LM 로드(get_lm) 및 JSON 형식 호출(invoke)."""

import json
import os

import dspy
from dotenv import load_dotenv

from src.translation.signatures.invoke import Invoke

load_dotenv()

_DEFAULT_MODEL = "gemini/gemini-2.0-flash"
_lm = None
_current_model = _DEFAULT_MODEL


def get_lm(
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
):
    """DSPy LM을 로드하고 전역으로 설정한다. 기본: gemini/gemini-2.0-flash."""
    global _lm, _current_model
    model = model or _DEFAULT_MODEL
    _current_model = model
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    _lm = dspy.LM(model, api_key=api_key, **kwargs)
    dspy.configure(lm=_lm)
    return _lm


def invoke(
    prompt: str,
    temperature: float | None = None,
    model: str | None = None,
) -> dict:
    """프롬프트를 LM에 보내고, JSON 형식 응답을 파싱해 dict로 반환한다."""
    if dspy.settings.lm is None:
        get_lm(model=model, temperature=temperature)
    elif model is not None or temperature is not None:
        get_lm(model=model or _current_model, temperature=temperature)

    predictor = dspy.Predict(Invoke)
    out = predictor(prompt=prompt)
    raw = getattr(out, "response", None) or str(out)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"text": raw}
