# Wissenschaft-AI

고서(독일어 철학/학술 텍스트) 기반 데이터를 한국어로 번역하기 위한 DSPy 실험 프로젝트입니다.  
핵심은 **번역 모듈 + LLM Judge 기반 점수화 + MIPROv2 프롬프트 최적화**입니다.

## 프로젝트 개요

- OCR/정제된 독일어 원문-번역 매핑 데이터를 기반으로 학습/검증 샘플을 구성합니다.
- `TranslateModule`이 독일어 원문을 한국어로 번역합니다.
- `TranslationQualityJudge`가 번역 품질을 다면 평가(5개 항목)합니다.
- `metric_llm`이 Judge 결과를 점수화해 최적화 지표로 사용합니다.
- `dspy.MIPROv2`가 해당 지표를 기준으로 번역 프롬프트를 최적화합니다.

## 전체 플로우

```text
[1] OCR / 데이터 준비
    - 원천: 고서 PDF
    - 결과: 원문-번역 매핑 JSON
      (예: resources/mapping_dataset/*.json, merged_mapping.json)

[2] 데이터 로딩
    src/translation/data/dataset.py
    - merged_mapping.json 로드
    - dspy.Example(original_text, translated_text) 변환
    - train/val 분할

[3] 번역 모듈
    src/translation/modules/translate.py
    - GermanToKorean Signature 기반 번역
    - 기본 LM 설정 후 추론

[4] 품질 평가 metric
    src/translation/signatures/translation_judge.py
    src/translation/metrics/translate_metric.py
    - faithfulness / terminology_accuracy / korean_fluency /
      style_register / overall_score 점수 산출
    - 최종 metric 점수 계산

[5] 프롬프트 최적화
    src/translation/optimizers/miprov2_optimizer.py
    - MIPROv2 compile 실행
    - artifacts/translation_optimized.json 저장

[6] 최적화 모델 추론/확인
    tests/test_translation/test_translate_monitoring.py
    - 저장된 optimized artifact 로드 후 단문 번역 실행
```

## 디렉터리 구조

```text
Wissenschaft-AI/
├── artifacts/                           # 최적화 결과 산출물(json)
├── data/                                # 실험용 데이터 폴더
├── logs/                                # 실행 로그
├── resources/
│   └── mapping_dataset/                 # 원문-번역 매핑 데이터
├── scripts/                             # 보조 스크립트/문서
├── src/
│   └── translation/
│       ├── data/dataset.py              # 데이터 로딩·분할
│       ├── signatures/                  # 번역/Judge 시그니처
│       ├── modules/translate.py         # 번역 모듈·LM 설정·추론
│       ├── metrics/translate_metric.py  # Judge 기반 metric
│       └── optimizers/miprov2_optimizer.py
└── tests/test_translation/
    └── test_translate_monitoring.py     # optimized 추론 확인
```

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

`.env`에 아래 값을 설정해야 번역/최적화가 동작합니다.

```env
ANTHROPIC_API_KEY=your_api_key
```

## 실행 가이드

### 1) MIPROv2 최적화 실행

```bash
python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2()"
```

완료되면 기본 경로 `artifacts/translation_optimized.json`이 생성됩니다.

### 2) 최적화된 모듈로 단문 번역 확인

```bash
python -m tests.test_translation.test_translate_monitoring
```

## Optimizer 로그 관리 규칙

최적화 실험 로그는 `logs/`에 누적 관리합니다.

- 3회 실험 결과 로그를 기준으로 관리:
  - `logs/log_1.txt`
  - `logs/log_2.txt`
  - `logs/log_3.txt`
- 로그에는 MIPROv2 실행 단계(bootstrap, instruction proposal, trial 등)와 점수 흐름이 포함됩니다.
- 추가 실험 시에도 같은 규칙으로 `log_4.txt`, `log_5.txt`처럼 번호를 이어서 저장하는 것을 권장합니다.

## `mapping_dataset` 구조 및 파싱 규칙

`resources/mapping_dataset/`는 챕터 단위 파일과 병합 파일을 함께 관리합니다.

- 챕터 단위 파일
  - `chapter1_mapping.json`
  - `chapter2_mapping.json`
  - `chapter3_mapping.json`
  - `introduction_mapping.json`
- 병합 파일
  - `merged_mapping.json`

파싱 시 아래 구조를 유지하세요.

- 인코딩: UTF-8
- 최상위 타입: JSON 배열(`list`)
- 각 원소 스키마:
  - `original` (독일어 원문, string)
  - `translation` (한국어 번역문, string)

즉, **향후 데이터 추가/전처리 시에도 위와 동일한 스키마와 파일 구조로 파싱/저장**해 두어야 `src/translation/data/dataset.py`에서 그대로 로드해 사용할 수 있습니다.

## `.ipynb`(OCR 전처리) 안내

루트의 `.ipynb`는 **고서 PDF OCR 전처리 실험 노트북**입니다.

- 본 프로젝트에서는 고서 특성(활자/스캔 품질/각주 구조)에서 인식률이 높아 `PaddleOCR`(`PPStructureV3`)을 사용했습니다.
- 다만 OCR 성능은 데이터 도메인에 크게 의존합니다.
  - 현대 문서, 표 중심 문서, 필기체, 저해상도 스캔 등은 다른 OCR이 더 유리할 수 있습니다.
  - 따라서 **반드시 본인 데이터셋으로 벤치마크 후 OCR 엔진을 선택**하는 것을 권장합니다.

## 참고

- `scripts/README.md`: 스크립트 폴더 용도 설명
- `resources/mapping_dataset/`: 매핑 데이터 원본/병합본 관리
