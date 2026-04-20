from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bmpt.tokenizer.loader import resolve_tokenizer_source


@dataclass(slots=True)
class Composer:
    name: str
    prompt_token_ids: list[torch.Tensor]
    pad_token_id: int
    max_total_len: int
    truncate_side: str = "left"
    pad_to_multiple_of: int | None = None
    output_pad_token_id: int | None = None

    def compose(
        self,
        outputs: list[torch.Tensor],
        output_masks: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(self.prompt_token_ids) != len(outputs) + 1:
            raise ValueError(
                f"Composer `{self.name}` expects {len(self.prompt_token_ids) - 1} outputs, got {len(outputs)}"
            )

        if len(outputs) == 0:
            raise ValueError(f"Composer `{self.name}` requires at least one output tensor")

        batch_size = int(outputs[0].size(0))
        device = outputs[0].device
        for output in outputs:
            if output.dim() != 2:
                raise ValueError("Each output tensor must be [B, T]")
            if int(output.size(0)) != batch_size:
                raise ValueError("All output tensors must share batch size")
            if output.device != device:
                raise ValueError("All output tensors must be on the same device")

        if output_masks is not None:
            if len(output_masks) != len(outputs):
                raise ValueError("output_masks length must match outputs length")
            for idx, mask in enumerate(output_masks):
                if mask.shape != outputs[idx].shape:
                    raise ValueError("Each output mask must have the same shape as its output tensor")
                if mask.device != device:
                    raise ValueError("All output masks must be on the same device as outputs")

        output_pad_token_id = self.pad_token_id if self.output_pad_token_id is None else self.output_pad_token_id
        prompt_lens = [int(prompt.numel()) for prompt in self.prompt_token_ids]
        output_lens: list[torch.Tensor] = []
        for idx, output in enumerate(outputs):
            if output_masks is not None:
                lens = output_masks[idx].to(dtype=torch.long).sum(dim=1)
            else:
                lens = (output != output_pad_token_id).to(dtype=torch.long).sum(dim=1)
            output_lens.append(lens)

        total_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        for idx in range(len(outputs)):
            total_lens = total_lens + prompt_lens[idx] + output_lens[idx]
        total_lens = total_lens + prompt_lens[-1]

        observed_max_len = int(total_lens.max().item())
        if self.max_total_len > 0:
            target_len = min(observed_max_len, self.max_total_len)
        else:
            target_len = observed_max_len

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 1:
            multiple = int(self.pad_to_multiple_of)
            target_len = ((target_len + multiple - 1) // multiple) * multiple
            if self.max_total_len > 0:
                target_len = min(target_len, self.max_total_len)

        if target_len <= 0:
            raise ValueError("target_len must be positive")

        input_ids = torch.full(
            (batch_size, target_len),
            fill_value=self.pad_token_id,
            dtype=outputs[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros((batch_size, target_len), dtype=torch.long, device=device)
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)

        prompts_on_device = [prompt.to(device=device, dtype=outputs[0].dtype) for prompt in self.prompt_token_ids]

        for sample_idx in range(batch_size):
            parts: list[torch.Tensor] = []
            for segment_idx, output in enumerate(outputs):
                parts.append(prompts_on_device[segment_idx])
                valid_len = int(output_lens[segment_idx][sample_idx].item())
                if valid_len > 0:
                    parts.append(output[sample_idx, :valid_len])
            parts.append(prompts_on_device[-1])

            merged = torch.cat(parts, dim=0)
            merged_len = int(merged.numel())
            if merged_len > target_len:
                if self.truncate_side == "left":
                    merged = merged[-target_len:]
                elif self.truncate_side == "right":
                    merged = merged[:target_len]
                else:
                    raise ValueError(f"Unsupported truncate_side: {self.truncate_side}")

            final_len = int(merged.numel())
            input_ids[sample_idx, :final_len] = merged
            attention_mask[sample_idx, :final_len] = 1
            lengths[sample_idx] = final_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
        }


def _load_tokenizer_for_prompting(config: dict[str, Any]) -> tuple[Any, int]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError("prompting composers require `transformers` for tokenizer loading") from exc

    local_source = config.get("prompting", {}).get("tokenizer_source")
    tokenizer_path = resolve_tokenizer_source(config, local_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.pad_token_id is None:
        raise ValueError("Failed to resolve pad_token_id for prompting tokenizer")
    return tokenizer, int(tokenizer.pad_token_id)


def _tokenize_prompt(
    tokenizer: Any,
    text: str,
    add_bos: bool,
    add_eos: bool,
) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = list(encoded["input_ids"])
    if add_bos and getattr(tokenizer, "bos_token_id", None) is not None:
        token_ids = [int(tokenizer.bos_token_id)] + token_ids
    if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        token_ids = token_ids + [int(tokenizer.eos_token_id)]
    return torch.tensor(token_ids, dtype=torch.long)


def build_composers_from_config(config: dict[str, Any]) -> dict[str, Composer]:
    prompting_cfg = config.get("prompting", {})
    composers_cfg = prompting_cfg.get("composers", {})
    if not isinstance(composers_cfg, dict) or len(composers_cfg) == 0:
        return {}

    tokenizer, default_pad_token_id = _load_tokenizer_for_prompting(config)
    composers: dict[str, Composer] = {}

    for name, composer_cfg_raw in composers_cfg.items():
        if not isinstance(composer_cfg_raw, dict):
            raise ValueError(f"prompting.composers.{name} must be a mapping")
        composer_cfg = dict(composer_cfg_raw)
        prompts = composer_cfg.get("prompts", [])
        if not isinstance(prompts, list) or len(prompts) < 2:
            raise ValueError(f"prompting.composers.{name}.prompts must contain at least 2 prompt segments")

        add_bos = bool(composer_cfg.get("add_bos", False))
        add_eos = bool(composer_cfg.get("add_eos", False))
        prompt_token_ids = [_tokenize_prompt(tokenizer, str(text), add_bos=add_bos, add_eos=add_eos) for text in prompts]

        output_pad_token_id = composer_cfg.get("output_pad_token_id")
        if output_pad_token_id is not None:
            output_pad_token_id = int(output_pad_token_id)

        composers[name] = Composer(
            name=name,
            prompt_token_ids=prompt_token_ids,
            pad_token_id=int(composer_cfg.get("pad_token_id", default_pad_token_id)),
            max_total_len=int(composer_cfg.get("max_total_len", 4096)),
            truncate_side=str(composer_cfg.get("truncate_side", "left")),
            pad_to_multiple_of=(
                None if composer_cfg.get("pad_to_multiple_of") is None else int(composer_cfg.get("pad_to_multiple_of"))
            ),
            output_pad_token_id=output_pad_token_id,
        )

    return composers
