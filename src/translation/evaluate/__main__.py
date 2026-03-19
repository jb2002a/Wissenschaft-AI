"""예: ``python -m src.translation.evaluate`` (프로젝트 루트에서, PYTHONPATH에 루트 포함)."""

import json
from datetime import datetime
from pathlib import Path

from src.translation.evaluate.test_json_eval import (
    SegmentEvalRow,
    TestJsonEvaluationReport,
    run_test_json_evaluation,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def report_to_plain_text(report: TestJsonEvaluationReport) -> str:
    """``TestJsonEvaluationReport`` / ``SegmentEvalRow`` 필드를 풀어 사람이 읽기 쉬운 텍스트로 만든다."""
    lines: list[str] = [
        f"json_path: {report.json_path}",
        f"use_optimized: {report.use_optimized}",
        f"system_score: {report.system_score:.6f}",
        f"num_segments: {len(report.segments)}",
        "",
    ]
    for i, seg in enumerate(report.segments):
        lines.extend(_segment_to_lines(i, seg))
    return "\n".join(lines).rstrip() + "\n"


def _segment_to_lines(index: int, seg: SegmentEvalRow) -> list[str]:
    block = [
        f"--- segment {index} ---",
        f"xcomet_score: {seg.xcomet_score:.6f}",
        f"german: {seg.german}",
        f"reference_korean: {seg.reference_korean}",
        f"hypothesis_korean: {seg.hypothesis_korean}",
    ]
    if seg.error_spans is not None:
        block.append(
            "error_spans:\n"
            + json.dumps(seg.error_spans, ensure_ascii=False, indent=2, default=str)
        )
    else:
        block.append("error_spans: null")
    block.append("")
    return block


def main() -> None:
    report = run_test_json_evaluation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        _REPO_ROOT / "logs" / "evaluation_logs" / f"evaluation_report_{timestamp}.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_to_plain_text(report), encoding="utf-8")
    print(f"저장됨: {out_path}")


if __name__ == "__main__":
    main()
