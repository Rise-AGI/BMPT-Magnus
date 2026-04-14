# VerifyBench Data Pipeline

## 1) Download raw VerifyBench (normal split file)

```bash
uv run python data/download_verifybench.py --repo-id ZJU-REAL/VerifyBench --filename verify_bench.jsonl --output-dir data/raw
```

Output:

- `data/raw/verify_bench.jsonl`

## 2) Prepare SFT training data

```bash
uv run python data/prepare_verifybench_sft.py --input data/raw/verify_bench.jsonl --output data/processed/train.jsonl --model Qwen/Qwen2.5-7B-Instruct --max-seq-len 4096
```

Output fields (JSONL per line):

- `prompt`
- `response`
- `input_ids`
- `attention_mask`
- `labels`

Notes:

- Only `verify_bench.jsonl` is used.
- `verify_bench_hard.jsonl` is intentionally not used.
- `labels` masks padded positions with `-100`.
