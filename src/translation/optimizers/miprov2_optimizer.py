"""MIPROv2를 사용한 번역 프로그램 compile 엔트리."""

# 직접 실행 예시:
# python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2()"
# python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2(save_path='artifacts/translation_optimized.json', load_if_exists=True, save_after_compile=True)"

import os
from typing import Any, Optional

import dspy

from src.translation.data.dataset import get_train_valset
from src.translation.metrics.translate_metric import metric_llm
from src.translation.modules.translate import TranslateModule, get_lm


def compile_translation_with_miprov2(
    train_ratio: float = 0.6,
    seed: int = 42,
    shuffle: bool = True,
    auto: str = "medium",
    max_bootstrapped_demos: int = 6,
    max_labeled_demos: int = 6,
    num_trials: int = 50,
    num_threads: Optional[int] = 4,
    save_path: Optional[str] = None,
    load_if_exists: bool = False,
    save_after_compile: bool = False,
    **compile_kwargs: Any,
) -> dspy.Module:
    """
    LM 설정 후 merged_mapping 길이 기준 train_ratio로 분할한 데이터로 MIPROv2를 실행한다.
    기본값은 항상 compile만 수행하고 저장/로드를 하지 않는다.
    load_if_exists=True면 save_path가 존재할 때 로드 후 반환한다.
    save_after_compile=True면 compile 결과를 save_path에 저장한다.
    """
    get_lm()
    if load_if_exists and not save_path:
        raise ValueError("load_if_exists=True이면 save_path를 함께 지정해야 합니다.")
    if save_after_compile and not save_path:
        raise ValueError("save_after_compile=True이면 save_path를 함께 지정해야 합니다.")

    if load_if_exists and save_path and os.path.exists(save_path):
        loaded = TranslateModule()
        loaded.load(save_path)
        return loaded

    trainset, valset = get_train_valset(train_ratio=train_ratio, seed=seed, shuffle=shuffle)

    optimizer = dspy.MIPROv2(
        metric=metric_llm,
        auto=auto,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_threads=num_threads,
    )

    student = TranslateModule()
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        num_trials=num_trials,
        **compile_kwargs,
    )

    if save_after_compile and save_path:
        optimized.save(save_path)

    return optimized
