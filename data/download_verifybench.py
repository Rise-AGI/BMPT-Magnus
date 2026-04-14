from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VerifyBench raw data")
    parser.add_argument("--repo-id", default="ZJU-REAL/VerifyBench")
    parser.add_argument("--filename", default="verify_bench.jsonl")
    parser.add_argument("--output-dir", default="data/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError("Please install `huggingface_hub` to download VerifyBench") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / args.filename

    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type="dataset",
    )
    shutil.copy2(downloaded_path, target_path)
    print(f"saved={target_path.resolve()}")


if __name__ == "__main__":
    main()
