# Wissenschaft-AI

extract(추출)과 translation(번역) 기능을 제공하는 프로젝트입니다.

## 폴더 구조

```
Wissenschaft-AI/
├── config/              # 프로젝트 설정
├── src/
│   ├── common/          # 공통 유틸·설정 (config 등)
│   ├── extract/         # 추출(extract) 기능
│   └── translation/     # 번역(translation) 기능
├── tests/
│   ├── test_extract/    # extract 단위·통합 테스트
│   └── test_translation/# translation 단위·통합 테스트
├── scripts/             # CLI·배치 실행 스크립트
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용

- **extract**: `src/extract/` — 텍스트·구조 추출 파이프라인
- **translation**: `src/translation/` — 번역 파이프라인

실행 예시는 `scripts/` 내 스크립트를 참고하세요.
