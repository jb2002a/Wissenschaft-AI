# Wissenschaft-AI

translation(번역) 기능을 제공하는 프로젝트입니다. DSPy 기반으로 동작하며, Judge 평가와 프롬프트 최적화를 지원합니다.

## 폴더 구조

```
Wissenschaft-AI/
├── config/                  # 프로젝트 설정
├── data/
│   ├── train/               # 최적화용 학습 데이터
│   └── eval/                # 평가용 데이터
├── resources/               # 원본 리소스 (PDF 등)
├── scripts/                 # CLI·배치 실행 스크립트
├── src/
│   └── translation/         # 번역(translation) 기능
│       ├── data/            # 데이터 로딩·dspy.Example 변환
│       ├── signatures/      # DSPy Signature (입출력 계약)
│       ├── modules/         # DSPy Module (번역·Judge 실행)
│       ├── metrics/         # Judge 기반 품질 metric
│       ├── optimizers/      # 프롬프트 최적화 (MIPRO 등)
│       └── pipeline.py      # 번역·최적화 오케스트레이션
├── tests/
│   └── test_translation/    # translation 단위·통합 테스트
├── requirements.txt
└── README.md
```

## 번역 파이프라인 (데이터 흐름)

```
data/train, data/eval
        ↓
translation/data (로딩·Example 변환)
        ↓
signatures (입출력 계약) → modules (번역·Judge 실행)
        ↓
metrics (Judge로 점수 계산)
        ↓
optimizers (점수 기반 프롬프트 최적화)
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용

- **translation**: `src/translation/` — DSPy 기반 번역 파이프라인 (Judge·최적화 포함)
  - 전체 실행: `src/translation/pipeline.py`의 `run_translate`, `run_optimize`
  - 실제 진입점: `scripts/` 내 스크립트에서 pipeline 호출

실행 예시는 `scripts/` 내 스크립트를 참고하세요.
