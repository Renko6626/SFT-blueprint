from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback

LOGGER = logging.getLogger("sft")


class GpuMemoryCallback(TrainerCallback):
    """Log GPU memory usage at every logging step and peak usage at training end.

    In multi-GPU training each process logs its own device. Under DDP all ranks
    have similar usage; under ZeRO-3 usage varies by rank, making per-rank logs
    useful for diagnosing memory imbalance.
    """

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        import torch

        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
        LOGGER.info(
            "GPU %d memory — allocated: %.2f GB | peak: %.2f GB | reserved: %.2f GB",
            device,
            allocated,
            peak,
            reserved,
        )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch

        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        LOGGER.info("GPU %d peak memory across full training run: %.2f GB", device, peak)
