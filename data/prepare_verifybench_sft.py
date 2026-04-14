from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Prepare VerifyBench for SFT")
    parser.add_argument("--input", default="data/raw/verify_bench.jsonl")
    parser.add_argument("--output", default="data/processed/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-len", type=int, default=4096)
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


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError("Please install `transformers` for SFT preprocessing") from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total = 0
    kept = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
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
            text = prompt + response

            encoded = tokenizer(
                text,
                max_length=int(args.max_seq_len),
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, -100)

            sample = {
                "prompt": prompt,
                "response": response,
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "labels": labels.tolist(),
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

    print(f"input={input_path.resolve()} total={total} kept={kept}")
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
