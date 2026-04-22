from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pbv_common import _safe_temp, _to_single_batch_tensor


def _sample_with_logprob(
    model: nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    require_grad: bool = False,
) -> tuple[list[int], torch.Tensor]:
    if not prompt_ids:
        prompt_ids = [0]

    input_ids = _to_single_batch_tensor(prompt_ids, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    do_sample = temperature > 0
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "return_dict_in_generate": True,
        "output_scores": not require_grad,
    }
    if do_sample:
        generate_kwargs["temperature"] = _safe_temp(temperature)

    with torch.no_grad():
        generated = model.generate(**generate_kwargs)

    generated_ids = generated["sequences"][0].tolist()
    new_ids = generated_ids[len(prompt_ids) :]

    if not new_ids:
        return new_ids, torch.zeros((), device=device, requires_grad=require_grad)

    if require_grad:
        full_ids = prompt_ids + new_ids
        full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        outputs = model(full_input_ids)
        logits = outputs.logits[:, len(prompt_ids) - 1 : len(full_ids) - 1, :]
        logprobs = torch.log_softmax(logits[0], dim=-1)
        total_logprob = torch.zeros((), device=device)
        for idx, token_id in enumerate(new_ids):
            total_logprob = total_logprob + logprobs[idx, token_id]
    else:
        if "scores" in generated and len(generated["scores"]) > 0:
            scores_stack = torch.stack(generated["scores"], dim=1)
            logprobs = torch.log_softmax(scores_stack[0], dim=-1)
            total_logprob = torch.zeros((), device=device)
            for idx, token_id in enumerate(new_ids):
                if idx < logprobs.size(0):
                    total_logprob = total_logprob + logprobs[idx, token_id]
        else:
            total_logprob = torch.zeros((), device=device)

    return new_ids, total_logprob


def _sample_with_logprob_batch(
    model: nn.Module,
    tokenizer: Any,
    prompt_ids: list[int],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    require_grad: bool = False,
) -> list[tuple[list[int], torch.Tensor]]:
    if not prompt_ids:
        prompt_ids = [0]

    do_sample = temperature > 0
    prompt_len = len(prompt_ids)

    results: list[tuple[list[int], torch.Tensor]] = []

    if require_grad:
        base_input_ids = _to_single_batch_tensor(prompt_ids, device=device)
        base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long)
        for _ in range(num_samples):
            generate_kwargs: dict[str, Any] = {
                "input_ids": base_input_ids,
                "attention_mask": base_attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": int(tokenizer.pad_token_id),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "return_dict_in_generate": True,
                "output_scores": False,
            }
            if do_sample:
                generate_kwargs["temperature"] = _safe_temp(temperature)

            with torch.no_grad():
                generated = model.generate(**generate_kwargs)

            gen_ids = generated["sequences"][0].tolist()
            new_ids = gen_ids[prompt_len:]

            if not new_ids:
                results.append((new_ids, torch.zeros((), device=device, requires_grad=True)))
                continue

            full_ids = prompt_ids + new_ids
            full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            outputs = model(full_input_ids)
            logits = outputs.logits[:, prompt_len - 1 : len(full_ids) - 1, :]
            logprobs = torch.log_softmax(logits[0], dim=-1)
            total_logprob = torch.zeros((), device=device)
            for idx, token_id in enumerate(new_ids):
                total_logprob = total_logprob + logprobs[idx, token_id]

            results.append((new_ids, total_logprob))

        return results

    input_ids = torch.tensor([prompt_ids] * num_samples, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = _safe_temp(temperature)

    with torch.no_grad():
        generated = model.generate(**generate_kwargs)

    sequences = generated["sequences"]
    scores = generated.get("scores", [])

    for i in range(num_samples):
        gen_ids = sequences[i].tolist()
        new_ids = gen_ids[prompt_len:]

        if not new_ids:
            results.append((new_ids, torch.zeros((), device=device, requires_grad=require_grad)))
            continue

        if require_grad:
            full_ids = prompt_ids + new_ids
            full_input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            outputs = model(full_input_ids)
            logits = outputs.logits[:, prompt_len - 1 : len(full_ids) - 1, :]
            logprobs = torch.log_softmax(logits[0], dim=-1)
            total_logprob = torch.zeros((), device=device)
            for idx, token_id in enumerate(new_ids):
                total_logprob = total_logprob + logprobs[idx, token_id]
        else:
            if scores and len(scores) > 0:
                scores_stack = torch.stack(scores, dim=1)
                logprobs = torch.log_softmax(scores_stack[i], dim=-1)
                total_logprob = torch.zeros((), device=device)
                for idx, token_id in enumerate(new_ids):
                    if idx < logprobs.size(0):
                        total_logprob = total_logprob + logprobs[idx, token_id]
            else:
                total_logprob = torch.zeros((), device=device)

        results.append((new_ids, total_logprob))

    return results


def _completion_logprob_batch(
    model: nn.Module,
    prompt_ids_list: list[list[int]],
    completion_ids_list: list[list[int]],
    device: torch.device,
) -> list[torch.Tensor]:
    if not prompt_ids_list or not completion_ids_list:
        return []

    results: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
            if not completion_ids:
                results.append(torch.zeros((), device=device))
                continue

            context_ids = prompt_ids.copy() if prompt_ids else [0]
            total_logprob = torch.zeros((), device=device)

            for token_id in completion_ids:
                input_ids = _to_single_batch_tensor(context_ids, device=device)
                outputs = model(input_ids)
                logits = outputs.logits[0, -1]
                logprobs = torch.log_softmax(logits, dim=-1)
                total_logprob = total_logprob + logprobs[token_id]
                context_ids.append(token_id)

            results.append(total_logprob)

    return results


def _completion_logprob(
    model: nn.Module,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    if not completion_ids:
        return torch.zeros((), device=device)

    context_ids = prompt_ids.copy() if prompt_ids else [0]
    total_logprob = torch.zeros((), device=device)

    model.eval()
    with torch.no_grad():
        for token_id in completion_ids:
            input_ids = _to_single_batch_tensor(context_ids, device=device)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]
            logprobs = torch.log_softmax(logits, dim=-1)
            total_logprob = total_logprob + logprobs[token_id]
            context_ids.append(token_id)

    return total_logprob
