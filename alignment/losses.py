"""Causally shifted response sums, token-mean SFT, and stable sequence DPO."""
import math
import torch
import torch.nn.functional as F
from .batching import build_sft_batch, build_dpo_batch


def model_device(model):
    return next(model.parameters(), torch.empty(0)).device


def response_logps(model, batch):
    """Return summed, causally shifted response-token log probabilities.

    The token at position zero has no preceding model position and therefore
    cannot be scored. For an empty prompt this means the first response token is
    intentionally unscoreable; later response tokens, including EOS when it
    survives truncation, are scored normally. A fully truncated response sums
    to scalar zero without NaN. PAD==EOS is safe because scoring uses attention
    and label masks. Rows retain batch order. Training data uses non-empty
    formatted prompts, so this edge case does not affect recorded experiments.
    """
    batch = {k: v.to(model_device(model)) for k, v in batch.items()}
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100) & batch["attention_mask"][:, 1:].bool()
    targets = labels.masked_fill(~mask, 0)
    # FP32 softmax under autocast; preserve FP64 for numerical checks.
    logits = logits[:, :-1]
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_fill(~mask, 0).sum(-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return response_logps(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    count = batch["labels"][:, 1:].ne(-100).sum().clamp_min(1)
    scores = response_logps(model, batch)
    return -scores.sum() / count.to(scores.device)


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    values = (policy_chosen_logps, policy_rejected_logps,
              reference_chosen_logps, reference_rejected_logps)
    if any(v.ndim != 1 or v.shape != values[0].shape for v in values) or not values[0].numel():
        raise ValueError("log probabilities must be nonempty equal-length vectors")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    pc, pr = values[:2]
    rc, rr = (v.detach().to(pc.device) for v in values[2:])
    return -F.logsigmoid(beta * ((pc - pr) - (rc - rr))).mean()


def freeze_reference(reference):
    reference.requires_grad_(False)
    reference.eval()


def dpo_scores(policy, reference, batch):
    if policy is reference or {id(p) for p in policy.parameters()} & {id(p) for p in reference.parameters()}:
        raise ValueError("policy and reference must not share parameters")
    freeze_reference(reference)
    pc = response_logps(policy, batch["chosen"])
    pr = response_logps(policy, batch["rejected"])
    with torch.no_grad():
        rc = response_logps(reference, batch["chosen"]).to(pc.device)
        rr = response_logps(reference, batch["rejected"]).to(pc.device)
    return pc, pr, rc, rr


def dpo_metrics(pc, pr, rc, rr, beta):
    chosen = beta * (pc.detach() - rc.detach())
    rejected = beta * (pr.detach() - rr.detach())
    return {"chosen_reward": chosen.mean().item(), "rejected_reward": rejected.mean().item(),
            "reward_accuracy": (chosen > rejected).float().mean().item(),
            "reward_margin": (chosen - rejected).mean().item()}


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return compute_dpo_loss_from_logps(*dpo_scores(policy_model, reference_model, batch), beta)
