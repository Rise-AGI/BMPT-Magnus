from __future__ import annotations

from typing import Any

import torch
from torch import nn

from bmpt.util import Composer

from pbv_common import _safe_temp, _to_single_batch_tensor


class QwenProcessVerifier(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_new_tokens: int = 6,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max(max_new_tokens, 1)
        self.temperature = max(float(temperature), 0.0)

    def _decode_judgement(self, text: str) -> float:
        normalized = text.strip().lower()
        if normalized.startswith("right"):
            return 1.0
        if normalized.startswith("wrong"):
            return 0.0
        return 0.0

    def judge_ids(
        self,
        prompt_ids: list[int],
        plan_step_ids: list[int],
        prefix_ids: list[int],
        candidate_ids: list[int],
        composer: Composer,
        device: torch.device,
    ) -> float:
        composed = composer.compose(
            outputs=[
                _to_single_batch_tensor(prompt_ids, device=device),
                _to_single_batch_tensor(plan_step_ids, device=device),
                _to_single_batch_tensor(prefix_ids, device=device),
                _to_single_batch_tensor(candidate_ids, device=device),
            ]
        )

        input_ids = composed["input_ids"]
        attention_mask = composed["attention_mask"]
        prompt_len = int(composed["lengths"][0].item())
        do_sample = self.temperature > 0
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=_safe_temp(self.temperature),
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        generated_tail = generated[0, prompt_len:]
        text = self.tokenizer.decode(generated_tail.tolist(), skip_special_tokens=True)
        return self._decode_judgement(text)

    def _pad_batch(
        self,
        composed_list: list[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        max_len = max(int(c["lengths"][0].item()) for c in composed_list)
        batch_size = len(composed_list)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        prompt_lens = []
        for i, c in enumerate(composed_list):
            seq_len = int(c["lengths"][0].item())
            input_ids[i, :seq_len] = c["input_ids"][0, :seq_len]
            attention_mask[i, :seq_len] = c["attention_mask"][0, :seq_len]
            prompt_lens.append(seq_len)
        return input_ids, attention_mask, prompt_lens

    def judge_ids_batch(
        self,
        prompt_ids_list: list[list[int]],
        plan_step_ids_list: list[list[int]],
        prefix_ids_list: list[list[int]],
        candidate_ids_list: list[list[int]],
        composer: Composer,
        device: torch.device,
    ) -> list[float]:
        batch_size = len(prompt_ids_list)
        if batch_size == 0:
            return []
        all_outputs = []
        for i in range(batch_size):
            composed = composer.compose(
                outputs=[
                    _to_single_batch_tensor(prompt_ids_list[i], device=device),
                    _to_single_batch_tensor(plan_step_ids_list[i], device=device),
                    _to_single_batch_tensor(prefix_ids_list[i], device=device),
                    _to_single_batch_tensor(candidate_ids_list[i], device=device),
                ]
            )
            all_outputs.append(composed)
        input_ids_batch, attention_mask_batch, prompt_lens = self._pad_batch(all_outputs, device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=_safe_temp(self.temperature),
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        scores = []
        for i in range(batch_size):
            prompt_len = prompt_lens[i]
            generated_tail = generated[i, prompt_len:]
            text = self.tokenizer.decode(generated_tail.tolist(), skip_special_tokens=True)
            scores.append(self._decode_judgement(text))
        return scores
