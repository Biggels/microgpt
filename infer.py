"""
Inference entrypoint for MicroGPT.

Loads a checkpoint and samples text from the model.
"""

import argparse
import random
from typing import List

from io_utils import load_checkpoint
from model import gpt_forward, softmax, state_dict_from_floats


def sample_one(
    state_dict,
    uchars: List[str],
    bos_token_id: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    block_size: int,
    temperature: float,
) -> str:
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = bos_token_id
    sample_chars: List[str] = []

    for pos_id in range(block_size):
        logits = gpt_forward(
            token_id=token_id,
            pos_id=pos_id,
            keys=keys,
            values=values,
            state_dict=state_dict,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        scaled_logits = [l / temperature for l in logits]
        probs = softmax(scaled_logits)
        token_id = random.choices(
            range(len(uchars) + 1), weights=[p.data for p in probs]
        )[0]
        if token_id == bos_token_id:
            break
        sample_chars.append(uchars[token_id])

    return "".join(sample_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MicroGPT inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint.pkl",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ckpt = load_checkpoint(args.checkpoint)
    state_dict = state_dict_from_floats(ckpt["state_dict"])
    metadata = ckpt.get("metadata", {})
    cfg = metadata.get("config", {})

    uchars = metadata.get("uchars")
    bos_token_id = metadata.get("bos_token_id")

    if uchars is None or bos_token_id is None:
        raise ValueError("Checkpoint missing vocab metadata (uchars/bos_token_id).")

    temperature = (
        args.temperature
        if args.temperature is not None
        else cfg.get("temperature", 0.5)
    )
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else cfg.get("num_samples", 100)
    )
    block_size = args.max_len if args.max_len is not None else cfg.get("block_size", 16)

    n_layer = cfg.get("n_layer", 1)
    n_head = cfg.get("n_head", 4)
    n_embd = cfg.get("n_embd", 16)

    print("\n--- inference (samples) ---")
    for sample_idx in range(num_samples):
        text = sample_one(
            state_dict=state_dict,
            uchars=uchars,
            bos_token_id=bos_token_id,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            temperature=temperature,
        )
        print(f"sample {sample_idx + 1:2d}: {text}")


if __name__ == "__main__":
    main()
