from __future__ import annotations

from typing import Any


def _require_hf() -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "`bmpt.tokenizer.loader` requires `transformers`. "
            "Please install it in your environment."
        ) from exc
    return AutoTokenizer


def load_tokenizer(config: dict[str, Any]) -> Any:
    AutoTokenizer = _require_hf()
    model_cfg = config["models"]["policy"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_vocab_hash(tokenizer: Any) -> str:
    vocab = getattr(tokenizer, "vocab", None)
    if vocab is None:
        vocab = getattr(tokenizer, "encoder", None)
    if vocab is None:
        return "unknown"
    vocab_keys = sorted(vocab.keys())
    vocab_hash = hash(tuple(vocab_keys))
    return str(abs(vocab_hash))