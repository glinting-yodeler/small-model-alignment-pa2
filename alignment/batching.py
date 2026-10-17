"""Separate prompt/response encoding, right padding, EOS before truncation.

No implicit BOS or prompt template. Empty batches are invalid; empty sequences
get one masked padding position. A response already ending in EOS still receives
the contract's appended EOS. Training data removes terminal EOS upstream only
if explicitly configured; the generic adapter never rewrites text.
"""
import torch


def encode_row(tokenizer, prompt, response, max_length, max_prompt_length=None):
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompts and responses must be strings")
    p = list(tokenizer.encode(prompt, add_special_tokens=False))
    if max_prompt_length is not None:
        if not 0 < max_prompt_length < max_length:
            raise ValueError("prompt cap must be positive and below max_length")
        p = p[-max_prompt_length:]  # keep response delimiter and prompt suffix
    r = list(tokenizer.encode(response, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        r.append(eos)
    return {"input_ids": (p + r)[:max_length],
            "labels": ([-100] * len(p) + r)[:max_length]}


def pad_rows(tokenizer, rows):
    if not rows:
        raise ValueError("batch must not be empty")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0  # valid conventional vocabulary index; masked by position
    width = max(1, max(len(r["input_ids"]) for r in rows))
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        attention[i, :n] = 1
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Encode rows in order with response masking and positional right padding.

    EOS is appended before truncation. A prompt filling ``max_length`` leaves
    a valid zero-response-token row. PAD==EOS remains safe because attention
    and labels define padding; a tokenizer without EOS appends nothing.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompt/response counts differ")
    return pad_rows(tokenizer, [encode_row(tokenizer, p, r, max_length)
                                for p, r in zip(prompts, responses)])


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Build aligned chosen/rejected batches without reordering input rows.

    Both sides use the response masking, EOS-before-truncation, and positional
    padding contract of :func:`build_sft_batch` independently.
    """
    if len(prompts) != len(chosen_responses) or len(prompts) != len(rejected_responses):
        raise ValueError("preference counts differ")
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}
