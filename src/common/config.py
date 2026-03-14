"""공통 설정 (환경 변수, 기본값 등)."""

import os
from pathlib import Path

# 프로젝트 루트 기준 경로
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# API 키 등은 환경 변수에서 로드 권장
# os.getenv("GOOGLE_API_KEY")
# os.getenv("LANGSMITH_API_KEY")
