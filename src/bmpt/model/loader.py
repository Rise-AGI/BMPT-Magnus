from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.distributed as dist


def _require_hf() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        raise ImportError(
            "`bmpt.model.loader` requires `transformers` and `peft`. "
            "Please install them in your environment."
        ) from exc

    return AutoModelForCausalLM, LoraConfig, TaskType, get_peft_model


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _resolve_attn_implementation(config: dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime", {})
    explicit = runtime_cfg.get("attn_implementation")
    if explicit is not None:
        value = str(explicit).strip()
        if value:
            return value

    if bool(runtime_cfg.get("flash_attention", False)):
        return "flash_attention_2"
    return "auto"


def _load_with_attn(
    loader_cls: Any,
    model_path: str,
    requested_attn: str,
) -> tuple[Any, str, bool]:
    base_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    attn_value = requested_attn.strip()
    attn_lower = attn_value.lower()
    if attn_lower == "flash_attention":
        attn_value = "flash_attention_2"
        attn_lower = attn_value

    if attn_lower == "default":
        model = loader_cls.from_pretrained(model_path, **base_kwargs)
        return model, "default", False

    if attn_lower == "auto":
        try:
            model = loader_cls.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **base_kwargs,
            )
            return model, "flash_attention_2", False
        except Exception as exc:
            if _is_rank0():
                warnings.warn(
                    f"FlashAttention auto probe failed, fallback to default attention: {exc}",
                    RuntimeWarning,
                )
            model = loader_cls.from_pretrained(model_path, **base_kwargs)
            return model, "default", True

    try:
        model = loader_cls.from_pretrained(
            model_path,
            attn_implementation=attn_value,
            **base_kwargs,
        )
        return model, attn_value, False
    except Exception as exc:
        if attn_lower == "flash_attention_2":
            if _is_rank0():
                warnings.warn(
                    f"Requested flash_attention_2 unavailable, fallback to default attention: {exc}",
                    RuntimeWarning,
                )
            model = loader_cls.from_pretrained(model_path, **base_kwargs)
            return model, "default", True
        raise RuntimeError(f"Unsupported or unavailable attn_implementation={attn_value}: {exc}") from exc


def _apply_lora_if_needed(model: Any, spec: dict[str, Any]) -> Any:
    _, LoraConfig, TaskType, get_peft_model = _require_hf()
    lora_cfg = spec.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )
    return get_peft_model(model, config)


def load_model(label: str, spec: dict[str, Any], config: dict[str, Any]) -> torch.nn.Module:
    AutoModelForCausalLM, _, _, _ = _require_hf()
    requested_attn = _resolve_attn_implementation(config)
    model, actual_attn, did_fallback = _load_with_attn(
        loader_cls=AutoModelForCausalLM,
        model_path=spec["path"],
        requested_attn=requested_attn,
    )

    runtime_cfg = config.get("runtime", {})
    enable_gradient_ckpt = bool(runtime_cfg.get("gradient_checkpointing", False))
    if enable_gradient_ckpt:
        if _is_rank0():
            print("[\033[34m训练\033[0m] gradient checkpointing 激活")
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif _is_rank0():
            warnings.warn("gradient_checkpointing is enabled in config but model has no gradient_checkpointing_enable()", RuntimeWarning)

    model = _apply_lora_if_needed(model, spec)

    if _is_rank0():
        print(
            "[bmpt] model_init "
            f"requested_attn={requested_attn} "
            f"actual_attn={actual_attn} "
            f"fallback={int(did_fallback)} "
            f"gradient_checkpointing={int(enable_gradient_ckpt)}",
            flush=True,
        )

    trainable = bool(spec.get("trainable", False))
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    return model
