"""
Training entrypoint for MicroGPT.

This keeps the loop explicit and hackable while adding run folders,
logging, and checkpointing.
"""

import argparse
import random
from typing import Tuple

from config import TrainConfig, load_json, to_dict
from data import build_tokenizer, load_docs
from io_utils import log_metrics, make_run_dir, save_checkpoint, save_json, write_text
from model import (
    flatten_params,
    gpt_forward,
    init_state_dict,
    softmax,
    state_dict_to_floats,
)


def compute_loss_for_doc(
    doc: str,
    tokenizer,
    state_dict,
    cfg: TrainConfig,
) -> Tuple[float, object]:
    # Tokenize and add BOS on both ends
    tokens = [tokenizer.bos_token_id] + tokenizer.encode(doc) + [tokenizer.bos_token_id]
    n = min(cfg.block_size, len(tokens) - 1)

    keys = [[] for _ in range(cfg.n_layer)]
    values = [[] for _ in range(cfg.n_layer)]

    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt_forward(
            token_id=token_id,
            pos_id=pos_id,
            keys=keys,
            values=values,
            state_dict=state_dict,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)
    return loss.data, loss


def train(cfg: TrainConfig, run_name: str = None) -> str:
    random.seed(cfg.seed)

    # Load data + tokenizer
    docs = load_docs(cfg.dataset_path, seed=cfg.seed)
    tokenizer = build_tokenizer(docs)

    # Initialize parameters
    state_dict = init_state_dict(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        block_size=cfg.block_size,
    )
    params = flatten_params(state_dict)

    # Adam buffers
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    # Run directory + metadata
    run_dir = make_run_dir(name=run_name)
    save_json(f"{run_dir}/config.json", to_dict(cfg))
    save_json(
        f"{run_dir}/vocab.json",
        {"uchars": tokenizer.uchars, "bos_token_id": tokenizer.bos_token_id},
    )
    write_text(f"{run_dir}/notes.txt", "MicroGPT training run\n")

    # Training loop
    for step in range(cfg.num_steps):
        doc = docs[step % len(docs)]
        loss_value, loss = compute_loss_for_doc(doc, tokenizer, state_dict, cfg)

        # Backprop
        loss.backward()

        # Adam update
        lr_t = cfg.learning_rate * (1 - step / cfg.num_steps)
        for i, p in enumerate(params):
            m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * p.grad
            v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * p.grad**2
            m_hat = m[i] / (1 - cfg.beta1 ** (step + 1))
            v_hat = v[i] / (1 - cfg.beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + cfg.eps_adam)
            p.grad = 0

        if (step + 1) % cfg.log_every == 0 or step == 0:
            print(f"step {step + 1:4d} / {cfg.num_steps:4d} | loss {loss_value:.4f}")
            log_metrics(f"{run_dir}/metrics.csv", step + 1, loss_value, lr_t)

    # Save checkpoint
    checkpoint_path = f"{run_dir}/checkpoint.pkl"
    save_checkpoint(
        checkpoint_path,
        state_dict=state_dict_to_floats(state_dict),
        optimizer_state={
            "m": m,
            "v": v,
            "step": cfg.num_steps,
        },
        metadata={
            "config": to_dict(cfg),
            "uchars": tokenizer.uchars,
            "bos_token_id": tokenizer.bos_token_id,
        },
    )
    print(f"saved checkpoint: {checkpoint_path}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MicroGPT")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config) if args.config else TrainConfig()
    train(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
