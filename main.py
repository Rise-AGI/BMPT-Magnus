from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brisk training entrypoint")
    parser.add_argument(
        "--entry",
        choices=["weighted", "engine"],
        default="engine",
        help="Which executable flow to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.entry == "weighted":
        from examples.example_weighted_step import main as run_weighted_example

        run_weighted_example()
        return
    from examples.example_engine_loop import main as run_engine_example

    run_engine_example()


if __name__ == "__main__":
    main()
