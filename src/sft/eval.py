from __future__ import annotations

from typing import Any


def evaluate_if_available(trainer: Any, eval_dataset: Any | None) -> dict[str, float]:
    if eval_dataset is None:
        return {}
    metrics = trainer.evaluate()
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
