"""Markdown 파일을 텍스트로 변환하는 유틸리티."""

from pathlib import Path
import re


def md_to_text(
    path: str | Path,
    *,
    remove_line_breaks: bool = True,
    encoding: str = "utf-8",
) -> str:
    """Markdown 파일에서 텍스트를 추출한다.

    Args:
        path: Markdown 파일 경로 (.md, .markdown).
        remove_line_breaks: True이면 줄바꿈을 공백으로 바꾸고 연속 공백을 하나로 합친다.
        encoding: 파일 읽기 인코딩.

    Returns:
        추출된 전체 텍스트.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".md", ".markdown"):
        raise ValueError(f"Markdown 파일이 아닙니다: {path}")

    raw = path.read_text(encoding=encoding)
    if remove_line_breaks:
        raw = re.sub(r"\s+", " ", raw).strip()
    return raw


# resources ocr_result-18.36.md to text 테스트
if __name__ == "__main__":
    path = Path(__file__).resolve().parents[3] / "resources" / "translated-18.36.md"
    text = md_to_text(path)
    print(text)