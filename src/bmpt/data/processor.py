from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bmpt.tokenizer.loader import get_vocab_hash


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_required_keys(
    records: list[dict[str, Any]],
    required_keys: list[str],
    source_path: str,
) -> None:
    if not required_keys:
        return
    for idx, record in enumerate(records):
        missing = [key for key in required_keys if key not in record]
        if missing:
            raise ValueError(
                f"Missing required keys in {source_path} at line {idx + 1}: {missing}"
            )


def tokenize_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    tokenize_keys: list[str],
    max_seq_len: int,
) -> list[dict[str, Any]]:
    if not tokenize_keys:
        return records

    processed: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        for key in tokenize_keys:
            text = str(record.get(key, ""))
            encoded = tokenizer(
                text,
                max_length=max_seq_len,
                truncation=True,
                add_special_tokens=False,
            )
            new_record[f"{key}_input_ids"] = encoded["input_ids"]
        processed.append(new_record)
    return processed


def compute_cache_hash(
    source_path: str,
    required_keys: list[str],
    tokenize_keys: list[str],
    max_seq_len: int,
    vocab_hash: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_path.encode("utf-8"))
    hasher.update(json.dumps(sorted(required_keys)).encode("utf-8"))
    hasher.update(json.dumps(sorted(tokenize_keys)).encode("utf-8"))
    hasher.update(str(max_seq_len).encode("utf-8"))
    hasher.update(vocab_hash.encode("utf-8"))

    source_file = Path(source_path)
    if source_file.exists():
        stat = source_file.stat()
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))

    return hasher.hexdigest()[:16]


def get_cache_path(source_path: str, cache_dir: str | None) -> Path:
    source_file = Path(source_path)
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        return cache_root / f"{source_file.stem}.tokenized.jsonl"
    return source_file.parent / f"{source_file.stem}.tokenized.jsonl"


def load_cached_metadata(cache_path: Path) -> dict[str, Any] | None:
    meta_path = cache_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def save_cached_metadata(cache_path: Path, metadata: dict[str, Any]) -> None:
    meta_path = cache_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def process_source(
    source_config: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
    cache_dir: str | None = None,
    force_reprocess: bool = False,
) -> list[dict[str, Any]]:
    source_path = str(source_config["path"])
    required_keys = list(source_config.get("required_keys", []))
    tokenize_keys = list(source_config.get("tokenize_keys", []))

    cache_path = get_cache_path(source_path, cache_dir)
    vocab_hash = get_vocab_hash(tokenizer)
    current_hash = compute_cache_hash(
        source_path,
        required_keys,
        tokenize_keys,
        max_seq_len,
        vocab_hash,
    )

    if not force_reprocess and cache_path.exists():
        cached_meta = load_cached_metadata(cache_path)
        if cached_meta is not None and cached_meta.get("hash") == current_hash:
            return load_jsonl(cache_path)

    records = load_jsonl(source_path)
    validate_required_keys(records, required_keys, source_path)
    processed = tokenize_records(records, tokenizer, tokenize_keys, max_seq_len)

    save_jsonl(processed, cache_path)
    save_cached_metadata(cache_path, {
        "hash": current_hash,
        "source_path": source_path,
        "required_keys": required_keys,
        "tokenize_keys": tokenize_keys,
        "max_seq_len": max_seq_len,
    })

    return processed


def process_all_sources(
    config: dict[str, Any],
    tokenizer: Any,
    force_reprocess: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    data_cfg = config.get("data", {})
    sources = data_cfg.get("sources", [])
    if not sources:
        raise ValueError("config['data']['sources'] is empty or missing")

    max_seq_len = int(data_cfg.get("max_seq_len", 4096))
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir is not None:
        cache_dir = str(cache_dir)

    result: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        source_path = str(source["path"])
        source_name = source.get("name", source_path)
        records = process_source(
            source,
            tokenizer,
            max_seq_len,
            cache_dir,
            force_reprocess,
        )
        result[str(source_name)] = records

    return result