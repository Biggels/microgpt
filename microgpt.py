"""
Minimal single-file entrypoint for MicroGPT.

This preserves the original "single file" feel while delegating to the
modular training and inference scripts.
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MicroGPT entrypoint (train or infer)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--config", type=str, default=None, help="Path to config JSON"
    )
    train_parser.add_argument(
        "--run-name", type=str, default=None, help="Custom run name"
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint.pkl",
    )
    infer_parser.add_argument("--temperature", type=float, default=None)
    infer_parser.add_argument("--num-samples", type=int, default=None)
    infer_parser.add_argument("--max-len", type=int, default=None)
    infer_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        from train import main as train_main

        sys.argv = [sys.argv[0]]
        if args.config:
            sys.argv += ["--config", args.config]
        if args.run_name:
            sys.argv += ["--run-name", args.run_name]
        train_main()
        return

    if args.command == "infer":
        from infer import main as infer_main

        sys.argv = [sys.argv[0], "--checkpoint", args.checkpoint]
        if args.temperature is not None:
            sys.argv += ["--temperature", str(args.temperature)]
        if args.num_samples is not None:
            sys.argv += ["--num-samples", str(args.num_samples)]
        if args.max_len is not None:
            sys.argv += ["--max-len", str(args.max_len)]
        if args.seed is not None:
            sys.argv += ["--seed", str(args.seed)]
        infer_main()
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
