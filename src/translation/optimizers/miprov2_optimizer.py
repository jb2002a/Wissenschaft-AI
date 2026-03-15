"""MIPROv2를 사용한 번역 프로그램 compile 엔트리."""

# 직접 실행 예시:
# python -c "from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2; compile_translation_with_miprov2(save_path='artifacts/translation_optimized.json')"

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
    *,
    save_path: str,
    **compile_kwargs: Any,
) -> dspy.Module:
    """
    LM 설정 후 merged_mapping 길이 기준 train_ratio로 분할한 데이터로 MIPROv2를 실행한다.
    save_path에 최적화된 프로그램을 저장한다.
    """
    get_lm()
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

    optimized.save(save_path)

    return optimized
