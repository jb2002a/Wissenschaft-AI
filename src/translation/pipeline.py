"""번역·Judge·최적화 전체 오케스트레이션."""

# python -c "from src.translation.pipeline import run_translation_optimization; run_translation_optimization(save_path='artifacts/translation_optimized.json')"
# 필요 시 langtrace로 모니터링하려면: LANGTRACE_API_KEY 설정 후 langtrace_python_sdk의 langtrace.init(api_key=...)를 호출한다.

import os

from dotenv import load_dotenv

load_dotenv()

import dspy

from src.translation.modules.translate import TranslateModule, get_lm
from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2


def run_translation_optimization(
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
    *,
    save_path: str,
    **kwargs,
) -> dspy.Module:
    """save_path가 존재하면 로드하고, 없으면 MIPROv2로 최적화 후 save_path에 저장해 반환한다."""
    if not save_path:
        raise ValueError("save_path는 필수입니다. 저장 경로를 문자열로 전달하세요.")

    if os.path.exists(save_path):
        get_lm()
        loaded = TranslateModule()
        loaded.load(save_path)
        return loaded

    return compile_translation_with_miprov2(
        train_ratio=train_ratio,
        seed=seed,
        shuffle=shuffle,
        save_path=save_path,
        **kwargs,
    )
