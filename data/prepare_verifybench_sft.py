from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
QUESTION_KEYS = (
    "question",
    "query",
    "problem",
    "input",
    "prompt",
    "instruction",
)
ANSWER_KEYS = (
    "answer",
    "response",
    "solution",
    "output",
    "target",
    "label",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VerifyBench for SFT (text only)")
    parser.add_argument("--input", default="data/raw/verify_bench.jsonl")
    parser.add_argument("--train-output", default="data/processed/train.jsonl")
    parser.add_argument("--val-output", default="data/processed/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
def _pick_value(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
def _fallback_pick(record: dict[str, Any], excluded: set[str]) -> str | None:
    for key, value in record.items():
        if key in excluded:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
def _build_prompt(question: str) -> str:
    system = "You are a careful verification assistant. Solve the task and provide a clear final answer."
    return (
        "### System\n"
        f"{system}\n\n"
        "### User\n"
        f"{question}\n\n"
        "### Assistant\n"
    )
def _split_samples(samples: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio <= 0:
        return samples, []
    items = list(samples)
    random.Random(seed).shuffle(items)
    val_count = int(len(items) * val_ratio)
    if val_count <= 0 and len(items) > 0:
        val_count = 1
    val_samples = items[:val_count]
    train_samples = items[val_count:]
    if not train_samples and val_samples:
        train_samples = [val_samples.pop()]
    return train_samples, val_samples
def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    train_output_path = Path(args.train_output)
    val_output_path = Path(args.val_output)
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("`--val-ratio` must be in [0, 1)")
    total = 0
    kept = 0
    samples: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            row = line.strip()
            if not row:
                continue
            total += 1
            record = json.loads(row)
            question = _pick_value(record, QUESTION_KEYS)
            answer = _pick_value(record, ANSWER_KEYS)
            if question is None:
                question = _fallback_pick(record, set(ANSWER_KEYS))
            if answer is None:
                answer = _fallback_pick(record, set(QUESTION_KEYS))
            if question is None or answer is None:
                continue
            prompt = _build_prompt(question)
            response = answer
            sample = {
                "prompt": prompt,
                "response": response,
            }
            samples.append(sample)
            kept += 1
    train_samples, val_samples = _split_samples(
        samples=samples,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    _write_jsonl(train_output_path, train_samples)
    _write_jsonl(val_output_path, val_samples)
    print(f"input={input_path.resolve()} total={total} kept={kept}")
    print(f"train_output={train_output_path.resolve()} size={len(train_samples)}")
    print(f"val_output={val_output_path.resolve()} size={len(val_samples)}")
if __name__ == "__main__":
    main()