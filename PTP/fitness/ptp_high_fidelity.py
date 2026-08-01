from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)



@dataclass
class HighFidelityConfig:

    problem: str = "tsp"
    env_name: str = ""
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    generator_params: Dict[str, Any] = field(default_factory=dict)
    policy_name: str = ""
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    rollout_strategy: str = "auto"
    hf_steps: int = 200
    hf_epochs: int = 0
    hf_instances_per_epoch: int = 0
    train_problem_size: int = 20
    valid_problem_sizes: Sequence[int] = (100,)
    train_batch_size: int = 64
    pomo_size: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    alpha: float = 0.05
    precision: str = "32-true"
    device: str = "cuda"
    seed: int = 0
    num_validation_episodes: int = 128
    validation_batch_size: int = 64


def resolve_pomo_size(pomo_size: int | None, problem_size: int) -> int:

    if pomo_size is None:
        return int(problem_size)
    value = int(pomo_size)
    if value <= 0:
        logger.warning(
            "Invalid pomo_size=%s; falling back to problem_size=%d", pomo_size, problem_size
        )
        return int(problem_size)
    return value


def get_total_hf_train_steps(config: HighFidelityConfig) -> int:

    if config.hf_epochs > 0 and config.hf_instances_per_epoch > 0:
        batch_size = max(int(config.train_batch_size), 1)
        steps_per_epoch = math.ceil(config.hf_instances_per_epoch / batch_size)
        total_steps = config.hf_epochs * steps_per_epoch
        return max(int(total_steps), 1)

    return max(int(config.hf_steps), 1)


def get_hf_epoch_plan(config: HighFidelityConfig) -> Tuple[int, int]:

    if config.hf_epochs > 0 and config.hf_instances_per_epoch > 0:
        batch_size = max(int(config.train_batch_size), 1)
        steps_per_epoch = math.ceil(config.hf_instances_per_epoch / batch_size)
        return max(int(steps_per_epoch), 1), int(config.hf_epochs)

    return 0, 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
