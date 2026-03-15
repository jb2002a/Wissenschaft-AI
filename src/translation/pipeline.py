"""번역·Judge·최적화 전체 오케스트레이션."""

# python -c "from src.translation.pipeline import run_translation_optimization; run_translation_optimization()"
# 필요 시 langtrace로 모니터링하려면: LANGTRACE_API_KEY 설정 후 langtrace_python_sdk의 langtrace.init(api_key=...)를 호출한다.

from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import dspy

from src.translation.optimizers.miprov2_optimizer import compile_translation_with_miprov2


def run_translation_optimization(
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
) -> dspy.Module:
    """MIPROv2로 번역 모듈을 최적화하고 반환한다. 데이터는 merged_mapping 길이 기준 train_ratio(기본 4:1)로 분할한다."""
    return compile_translation_with_miprov2(
        train_ratio=train_ratio,
        seed=seed,
        shuffle=shuffle,
        save_path=save_path,
        **kwargs,
    )
