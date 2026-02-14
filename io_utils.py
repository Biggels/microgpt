"""
Checkpoint and run-folder utilities for MicroGPT.

This module is intentionally small and explicit to keep experimentation easy.
"""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_dir(root: str = "runs", name: Optional[str] = None) -> str:
    """
    Create and return a new run directory.

    If name is not provided, a timestamp-based name is used.
    """
    ensure_dir(root)
    if name is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        name = timestamp
    run_dir = os.path.join(root, name)
    ensure_dir(run_dir)
    return run_dir


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    path: str,
    state_dict: Dict[str, List[List[float]]],
    optimizer_state: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Save a checkpoint.

    state_dict should be numeric (floats), not Value objects.
    optimizer_state can include Adam buffers and step.
    metadata can include config, vocab, etc.
    """
    payload = {
        "state_dict": state_dict,
        "optimizer_state": optimizer_state,
        "metadata": metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_metrics_header(path: str) -> None:
    if os.path.exists(path):
        return
    write_text(path, "step,loss,lr\n")


def log_metrics(path: str, step: int, loss: float, lr: float) -> None:
    write_metrics_header(path)
    append_text(path, f"{step},{loss:.6f},{lr:.8f}\n")
