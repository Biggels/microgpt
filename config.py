"""
Configuration utilities for MicroGPT experiments.

This module keeps config handling minimal and explicit so experimentation
remains straightforward and hackable.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "inputs/phrases.txt"
    seed: int = 42

    # Model
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 16

    # Optimizer
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8

    # Training
    num_steps: int = 1000
    log_every: int = 10

    # Sampling
    temperature: float = 0.5
    num_samples: int = 100


def to_dict(cfg: TrainConfig) -> Dict[str, Any]:
    return asdict(cfg)


def from_dict(data: Dict[str, Any]) -> TrainConfig:
    return TrainConfig(**data)


def save_json(cfg: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(cfg), f, indent=2, sort_keys=True)


def load_json(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data)
