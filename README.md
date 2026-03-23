# Wissenschaft-AI

고서(독일어 철학·학술 텍스트)를 한국어로 옮기는 **실험 프로젝트**입니다.  
**XCOMET 계열 평가**로 번역 품질을 정량화하고, 그 결과를 바탕으로 **DSPy 프롬프트·데모를 개선**, **멀티 에이전트(역할 분리·다단계 추론 등) 설계** 등의 여러 기법들을 적용해보며, 최선의 결과를 내는것을 목표로 합니다.

## 전체 플로우

```text
[1] OCR / 데이터 준비
    - 원천: 고서 PDF 등
    - 학습용: 원문–참조 한국어 매핑 JSON
      (예: resources/mapping_dataset/*.json, merged_mapping.json)
    - 평가용: german / korean 필드를 가진 test.json
      (예: resources/test_dataset/test.json)

[2] 데이터 로딩 (학습·검증)
    src/translation/data/dataset.py
    - merged_mapping.json → dspy.Example(original_text, translated_text)
    - train / val 분할

[3] 번역 모듈 (DSPy student)
    src/translation/modules/translate.py
    - GermanToKorean 시그니처, TranslateModule
    - Gemini 등 LM은 get_lm()에서 설정

[4] 프롬프트 최적화 (기본 경로)
    src/translation/optimizers/miprov2_optimizer.py
    - MIPROv2 + src/translation/metrics/translate_metric_comet.py (wmt22-comet-da)
    - artifacts/translation_optimized.json 등에 저장

[5] XCOMET-XL 평가 (Modal)
    src/translation/evaluate/xcomet_modal_app.py
    src/translation/evaluate/test_json_eval.py
    - 번역 가설 생성 → 원격 XCOMET-XL predict → 리포트·로그

[6] (레거시) LLM Judge metric으로 MIPROv2
    src/translation/optimizers/optimizer_llm_judge.py
    src/translation/metrics/translate_metric_llm.py
    src/translation/signatures/legacy/translation_judge_legacy.py

[7] 스모크 / 모니터링
    tests/test_translation/test_translate_monitoring.py
    - 저장된 artifact로 긴 원문 번역 등 확인
```

## 디렉터리 구조

```text
Wissenschaft-AI/
├── artifacts/                    # MIPROv2 compile 산출물(JSON)
├── data/
├── logs/                         # 최적화 로그, evaluation_logs 등
├── resources/
│   ├── mapping_dataset/          # 학습·검증용 매핑
│   └── test_dataset/             # XCOMET 평가용 test.json
├── scripts/
├── src/translation/
│   ├── data/dataset.py
│   ├── evaluate/
│   │   ├── xcomet_modal_app.py   # Modal + XCOMET-XL 원격 채점
│   │   └── test_json_eval.py     # test.json 파이프라인·리포트
│   ├── signatures/
│   │   ├── german_to_korean.py
│   │   └── legacy/               # Judge 시그니처(레거시)
│   ├── modules/translate.py
│   ├── metrics/
│   │   ├── translate_metric_comet.py   # MIPROv2용 COMET-DA
│   │   └── translate_metric_llm.py     # 레거시 LLM Judge metric
│   └── optimizers/
│       ├── miprov2_optimizer.py        # COMET metric (권장)
│       └── optimizer_llm_judge.py      # LLM Judge metric (레거시)
└── tests/test_translation/
```

## 설치

```bash
pip install -r requirements.txt
```

`unbabel-comet`는 로컬 COMET-DA metric에 사용됩니다. Modal 기반 XCOMET-XL 평가는 [Modal](https://modal.com) CLI 설정과 별도 시크릿이 필요합니다.

## 환경 변수

| 용도 | 변수 |
|------|------|
| DSPy 번역·MIPROv2(기본 LM) | `GOOGLE_API_KEY` — `translate.py`의 `get_lm()`에서 Gemini 사용 |
| Modal XCOMET | Modal 앱에서 `huggingface` 시크릿 등(HF 토큰) — `xcomet_modal_app.py` 참고 |

일부 테스트 스크립트는 예전 키 이름을 참조할 수 있으니, 실행 전 해당 파일의 조건을 확인하세요.

## 실행 가이드

### 1) MIPROv2 최적화 (COMET-DA metric)

```bash
python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2()"
```

완료 후 기본 경로 `artifacts/translation_optimized.json`이 생성됩니다.

### 2) XCOMET-XL 평가 (Modal)

프로젝트 루트에서 Modal이 설정된 상태로:

```bash
modal run src/translation/evaluate/xcomet_modal_app.py
```

`test_json_eval.run_test_json_evaluation`이 번역 artifact를 적용한 뒤 원격 채점하고, `logs/evaluation_logs/`에 리포트를 씁니다.

### 3) 최적화 모듈로 번역 확인

```bash
python -m tests.test_translation.test_translate_monitoring
```

## Optimizer 로그 관리 규칙

최적화 실험 로그는 `logs/`에 누적 관리합니다.

- 예: `logs/log_1.txt`, `logs/log_2.txt`, …
- MIPROv2 단계(bootstrap, instruction proposal, trial 등)와 점수 흐름을 남깁니다.
- XCOMET 평가는 `logs/evaluation_logs/` 쪽 리포트와 함께 관리하는 것을 권장합니다.

## `mapping_dataset` 스키마

- 인코딩: UTF-8  
- 최상위: JSON 배열  
- 각 원소: `original`(독일어), `translation`(한국어 참조)

데이터 추가 시에도 동일 스키마를 유지하면 `dataset.py` 로더와 COMET metric 입력 형식이 맞습니다.

## `test_dataset/test.json` 스키마

평가 파이프라인은 항목마다 최소 `german`, `korean` 키를 기대합니다. 번역 후 `hypothesis_korean`이 붙고 XCOMET 입력으로 변환됩니다.

## `.ipynb`(OCR 전처리) 안내

루트의 `.ipynb`는 고서 PDF OCR 전처리 실험용입니다. 도메인에 따라 OCR 엔진 선택은 별도 벤치마크 후 결정하는 것이 좋습니다.

## 참고

- `scripts/README.md`: 스크립트 폴더 설명
- `resources/mapping_dataset/`: 학습용 매핑
- `resources/test_dataset/`: XCOMET 평가용 JSON
