"""MIPROv2를 사용한 번역 프로그램 compile 엔트리."""

# 직접 실행 예시:
# python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2()"
# python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2(save_path='artifacts/translation_optimized.json')"

import os
from typing import Any, Optional

import dspy

from src.translation.data.dataset import get_train_valset
from src.translation.metrics.translate_metric import metric_llm
from src.translation.modules.translate import TranslateModule, get_lm


def compile_translation_with_miprov2(
    train_ratio: float = 0.5,
    seed: int = 42,
    shuffle: bool = True,
    auto: str = "medium",
    max_bootstrapped_demos: int = 6,
    max_labeled_demos: int = 6,
    num_threads: Optional[int] = 4,
    save_path: str = "artifacts/translation_optimized.json",
    save_after_compile: bool = True,
    **compile_kwargs: Any,
) -> None:
    """
    LM 설정 후 merged_mapping 길이 기준 train_ratio로 분할한 데이터로 MIPROv2를 실행한다.
    기본값은 compile 결과를 save_path에 저장하며, load/return 동작은 하지 않는다.
    save_after_compile=False면 compile만 수행하고 저장하지 않는다.
    """
    get_lm()
    if save_after_compile and not save_path.strip():
        raise ValueError("save_after_compile=True이면 save_path를 함께 지정해야 합니다.")

    trainset, valset = get_train_valset(train_ratio=train_ratio, seed=seed, shuffle=shuffle)

    optimizer = dspy.MIPROv2(
        metric=metric_llm,
        auto=auto,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_threads=num_threads,
    )

    # auto 설정 시 num_trials/num_candidates는 사용 불가(에러 방지)
    compile_kwargs = {k: v for k, v in compile_kwargs.items() if k not in ("num_trials", "num_candidates")}

    student = TranslateModule()
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
        **compile_kwargs,
    )

    if save_after_compile:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        optimized.save(save_path)
