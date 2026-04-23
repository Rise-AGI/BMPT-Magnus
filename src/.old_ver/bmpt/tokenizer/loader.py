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


def resolve_tokenizer_source(
    config: dict[str, Any],
    local_source: str | None = None,
) -> str:
    """通用tokenizer来源解析
    
    优先级：
    1. local_source（调用方传入，如从data/prompting提取）
    2. 顶层 tokenizer_source
    3. 默认 models.policy.path
    
    支持值：
    - models中的key（如"policy"/"reference") → 使用对应模型path
    - 直接路径 → 直接返回
    """
    model_cfg = config.get("models", {})
    
    if local_source is not None:
        if isinstance(local_source, str) and local_source in model_cfg:
            model_spec = model_cfg[local_source]
            if isinstance(model_spec, dict) and "path" in model_spec:
                return str(model_spec["path"])
        return str(local_source)
    
    tokenizer_source = config.get("tokenizer_source")
    if tokenizer_source is not None:
        if isinstance(tokenizer_source, str) and tokenizer_source in model_cfg:
            model_spec = model_cfg[tokenizer_source]
            if isinstance(model_spec, dict) and "path" in model_spec:
                return str(model_spec["path"])
        return str(tokenizer_source)
    
    policy_spec = model_cfg.get("policy")
    if isinstance(policy_spec, dict) and "path" in policy_spec:
        return str(policy_spec["path"])
    
    raise ValueError("Cannot resolve tokenizer_source: missing models.policy.path")


def load_tokenizer(config: dict[str, Any]) -> Any:
    AutoTokenizer = _require_hf()
    local_source = config.get("data", {}).get("tokenizer_source")
    tokenizer_path = resolve_tokenizer_source(config, local_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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