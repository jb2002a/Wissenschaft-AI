"""test.json 로드 → translate 모듈로 번역 → XCOMET-XL로 평가."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from src.translation.modules.translate import translate, translate_with_optimized

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TEST_JSON = _REPO_ROOT / "resources" / "test_dataset" / "test.json"
_XCOMET_XL = "Unbabel/XCOMET-XL"


@dataclass
class SegmentEvalRow:
    """한 세그먼트의 원문·참조·가설 번역 및 XCOMET 점수. 데이터 wrapper class"""

    german: str
    reference_korean: str
    hypothesis_korean: str
    xcomet_score: float
    error_spans: Optional[List[Any]] = None


@dataclass
class TestJsonEvaluationReport:
    """데이터셋 전체 평가 결과."""

    system_score: float
    segments: List[SegmentEvalRow] = field(default_factory=list)
    json_path: Path = field(default_factory=lambda: _DEFAULT_TEST_JSON)
    use_optimized: bool = False


def load_test_json(path: Optional[Path] = None) -> List[dict]:
    """test.json 형식: 각 항목에 ``german``, ``korean`` 키."""
    p = path or _DEFAULT_TEST_JSON
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def translate_test_items(
    items: List[dict],
    *,
    use_optimized: bool = False,
    optimized_path: str = "artifacts/translation_optimized.json",
) -> List[dict]:
    """
    각 항목의 ``german``을 ``translate`` / ``translate_with_optimized``로 번역해
    ``hypothesis_korean`` 필드를 추가한 복사본 리스트를 반환.
    """
    out: List[dict] = []
    for row in items:
        g = row["german"]
        if use_optimized:
            hyp = translate_with_optimized(g, optimized_path=optimized_path)
        else:
            hyp = translate(g)
        new_row = {**row, "hypothesis_korean": hyp}
        out.append(new_row)
    return out


def build_xcomet_samples(rows: List[dict]) -> List[dict]:
    """COMET predict용 dictionary 재구성"""
    return [
        {
            "src": r["german"],
            "mt": r["hypothesis_korean"],
            "ref": r["korean"],
        }
        for r in rows
    ]


def score_with_xcomet(
    samples: List[dict],
    *,
    model_name: str = _XCOMET_XL,
    batch_size: int = 8,
    gpus: int = 0,
):
    """
    XCOMET-XL(또는 동일 API의 다른 COMET 체크포인트)으로 점수 산출.

    ``gpus=0``이면 CPU 추론.

    ``unbabel-comet`` 패키지가 필요하며, 모듈 로드 시점이 아니라 이 함수 호출 시에만 임포트한다.
    """
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model.predict(samples, batch_size=batch_size, gpus=gpus)


def run_test_json_evaluation(
    json_path: Optional[Path] = None,
    *,
    use_optimized: bool = True,
    optimized_path: str = "artifacts/translation_optimized.json",
    batch_size: int = 8,
    gpus: int = 0,
    model_name: str = _XCOMET_XL,
) -> TestJsonEvaluationReport:
    """
    기본 경로의 test.json(또는 ``json_path``)에 대해 번역 후 XCOMET-XL 평가를 수행한다.
    """
    path = json_path or _DEFAULT_TEST_JSON
    raw = load_test_json(path)
    translated = translate_test_items(
        raw,
        use_optimized=use_optimized,
        optimized_path=optimized_path,
    )
    samples = build_xcomet_samples(translated)
    model_output = score_with_xcomet(
        samples,
        model_name=model_name,
        batch_size=batch_size,
        gpus=gpus,
    )

    error_spans_list = model_output.metadata.error_spans

    segment_rows: List[SegmentEvalRow] = []
    for i, row in enumerate(translated):
        segment_rows.append(
            SegmentEvalRow(
                german=row["german"],
                reference_korean=row["korean"],
                hypothesis_korean=row["hypothesis_korean"],
                xcomet_score=float(model_output.scores[i]),
                error_spans=error_spans_list[i],
            )
        )

    return TestJsonEvaluationReport(
        system_score=float(model_output.system_score),
        segments=segment_rows,
        json_path=path,
        use_optimized=use_optimized,
    )
