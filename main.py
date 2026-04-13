from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brisk training entrypoint")
    parser.add_argument(
        "--entry",
        choices=["weighted", "engine", "train"],
        default="train",
        help="Which executable flow to run",
    )
    parser.add_argument(
        "--config",
        default="train/config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--loader",
        default="train.default_components:load_model",
        help="Model loader symbol path module:function",
    )
    parser.add_argument(
        "--dataloader",
        default="train.default_components:build_dataloader",
        help="Dataloader builder symbol path module:function",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max training steps override",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "deepspeed"],
        default=None,
        help="Optional training backend override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entry == "weighted":
        from examples.example_weighted_step import main as run_weighted_example

        run_weighted_example()
        return

    if args.entry == "engine":
        from examples.example_engine_loop import main as run_engine_example

        run_engine_example()
        return

    from train.run_train import run_train

    run_train(
        config_path=args.config,
        loader=args.loader,
        dataloader=args.dataloader,
        max_steps_override=args.max_steps,
        backend_override=args.backend,
    )


if __name__ == "__main__":
    main()
