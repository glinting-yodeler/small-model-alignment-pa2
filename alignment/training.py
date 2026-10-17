"""Custom PyTorch engine shared by SFT and both independent DPO tracks."""
from contextlib import nullcontext
import math
from pathlib import Path
import random
import time
import warnings
import torch
from .batching import encode_row, pad_rows
from .checkpointing import save_state, load_state
from .formatting import format_prompt, TEMPLATE
from .io import digest, write_json, append_jsonl, environment, read_json
from .losses import response_logps, dpo_scores, compute_dpo_loss_from_logps, dpo_metrics, freeze_reference


def encode_records(tokenizer, records, config):
    result = []
    empty_formatted = 0
    for record in records:
        prompt = format_prompt(record["prompt"])
        if not tokenizer.encode(prompt, add_special_tokens=False):
            empty_formatted += 1
        sides = ("response",) if config["track"] == "sft" else ("chosen", "rejected")
        item = {side: encode_row(tokenizer, prompt, record[side], config["max_length"],
                                config.get("max_prompt_length")) for side in sides}
        if any(not any(t != -100 for t in row["labels"][1:]) for row in item.values()):
            raise ValueError(f"Unscoreable training record {record.get('id')}; prepare with tokenizer filtering")
        result.append(item["response"] if sides == ("response",) else item)
    if not result:
        raise ValueError("Empty training/validation split")
    if empty_formatted:
        warnings.warn(f"{empty_formatted} records have an empty tokenized formatted prompt; first response token is unscoreable",
                      RuntimeWarning, stacklevel=2)
    return result


def collate(tokenizer, rows, track):
    if track == "sft":
        return pad_rows(tokenizer, rows)
    return {side: pad_rows(tokenizer, [r[side] for r in rows]) for side in ("chosen", "rejected")}


def objective(model, reference, batch, config):
    if config["track"] == "sft":
        scores = response_logps(model, batch)
        n = int(batch["labels"][:, 1:].ne(-100).sum())
        return -scores.sum() / n, n, {}
    scores = dpo_scores(model, reference, batch)
    loss = compute_dpo_loss_from_logps(*scores, config["beta"])
    return loss, len(scores[0]), dpo_metrics(*scores, config["beta"])


def amp_context(config):
    return torch.autocast("cuda", dtype=torch.float16) if config.get("amp", True) and str(config["device"]).startswith("cuda") else nullcontext()


def validate(model, reference, tokenizer, rows, config):
    previous = model.training
    model.eval()
    totals, count = {}, 0
    try:
        with torch.no_grad(), amp_context(config):
            for start in range(0, len(rows), config["microbatch"]):
                batch = collate(tokenizer, rows[start:start + config["microbatch"]], config["track"])
                loss, n, metrics = objective(model, reference, batch, config)
                for k, v in {"loss": loss.item(), **metrics}.items():
                    totals[k] = totals.get(k, 0) + v * n
                count += n
    finally:
        model.train(previous)
    values = {k: v / count for k, v in totals.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise FloatingPointError("Nonfinite validation metric")
    return values


def export_model(model, tokenizer, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "alignment_checkpoint.json"
    marker.unlink(missing_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
    else:  # real engine exercised by tiny offline models
        torch.save(model.state_dict(), path / "toy_model.pt")
    write_json(marker, metadata)


def train(model, tokenizer, train_records, validation_records, config, reference=None,
          resume=None, stop_after=None):
    """stop_after is an interruption test hook; max_updates in config is a pilot budget."""
    config = dict(config)
    for key in ("microbatch", "accumulation", "epochs", "eval_every"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    model.to(config["device"], dtype=torch.float32)
    if reference is not None:
        freeze_reference(reference)
    if config["track"] != "sft":
        if reference is None:
            raise ValueError("DPO requires a frozen SFT reference")
        # Remove stochastic dropout so initial identical scores are meaningful.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    train_rows = encode_records(tokenizer, train_records, config)
    val_rows = encode_records(tokenizer, validation_records, config)
    out = Path(config["output"])
    identity = digest({"config": config, "train": train_records, "validation": validation_records,
                       "source_hash": environment()["source_hash"]})
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError(f"Run already exists: {out}; use --resume or a new output")
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    windows_per_epoch = math.ceil(len(train_rows) / (config["microbatch"] * config["accumulation"]))
    total = windows_per_epoch * config["epochs"]
    total = min(total, config.get("max_updates") or total)
    warmup = math.ceil(total * config.get("warmup_ratio", 0))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) if config.get("scheduler") == "cosine" else None
    scaler = torch.amp.GradScaler("cuda", enabled=config.get("amp", True) and str(config["device"]).startswith("cuda"))
    pos = {"epoch": 0, "offset": 0, "step": 0, "best": None, "attempts": 0}
    if resume:
        pos = load_state(resume, model, opt, scheduler, scaler, identity)
    write_json(out / "resolved_config.json", config)
    write_json(out / "manifest.json", {**environment(), "identity": identity, "template": TEMPLATE,
                                       "train_count": len(train_rows), "validation_count": len(val_rows)})
    started = time.monotonic()
    def evaluate_and_save():
        metrics = validate(model, reference, tokenizer, val_rows, config)
        key = [-metrics["loss"]] if config["track"] == "sft" else [metrics["reward_accuracy"], -metrics["loss"]]
        append_jsonl(out / "metrics.jsonl", {"phase": "validation", "step": pos["step"], **metrics})
        if pos["best"] is None or key > pos["best"]:
            pos["best"] = key
            if not config.get("metrics_only", False):
                export_model(model, tokenizer, out / "best", {"track": config["track"], "kind": "best",
                             "step": pos["step"], "identity": identity, "metrics": metrics, "template": TEMPLATE})
    if not resume:
        evaluate_and_save()
    model.train()
    while pos["epoch"] < config["epochs"] and pos["step"] < total:
        order = list(range(len(train_rows)))
        random.Random(config["seed"] + pos["epoch"]).shuffle(order)
        start = pos["offset"]
        end = min(len(order), start + config["microbatch"] * config["accumulation"])
        batches = [collate(tokenizer, [train_rows[i] for i in order[s:min(s + config["microbatch"], end)]], config["track"])
                   for s in range(start, end, config["microbatch"])]
        weights = [int(b["labels"][:, 1:].ne(-100).sum()) if config["track"] == "sft"
                   else b["chosen"]["input_ids"].shape[0] for b in batches]
        denominator = sum(weights)
        opt.zero_grad(set_to_none=True)
        totals = {}
        tick = time.monotonic()
        for batch, weight in zip(batches, weights):
            with amp_context(config):
                loss, _, metrics = objective(model, reference, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss * (weight / denominator)).backward()
            for k, v in {"loss": loss.item(), **metrics}.items():
                totals[k] = totals.get(k, 0) + v * weight / denominator
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"])
        if not scaler.is_enabled() and not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        old_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        success = scaler.get_scale() >= old_scale
        pos["attempts"] += 1
        if not success:
            if pos["attempts"] > total * 10:
                raise FloatingPointError("Repeated AMP overflows")
            continue  # retry same window, scheduler and sampler do not advance
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        if scheduler:
            scheduler.step()
        pos["step"] += 1
        pos["offset"] = end
        if end == len(order):
            pos["epoch"] += 1
            pos["offset"] = 0
        append_jsonl(out / "metrics.jsonl", {"phase": "train", "step": pos["step"], **totals,
                     "gradient_norm": float(norm), "learning_rate": opt.param_groups[0]["lr"],
                     "seconds": time.monotonic() - tick, "examples": end - start})
        at_end = pos["step"] == total or pos["epoch"] == config["epochs"]
        if pos["step"] % config["eval_every"] == 0 or pos["offset"] == 0 or at_end:
            evaluate_and_save()
        if (not config.get("metrics_only", False) and not config.get("skip_resume_state", False) and
                (pos["step"] % config.get("save_every", config["eval_every"]) == 0 or at_end or
                 (stop_after and pos["step"] >= stop_after))):
            save_state(out / "latest.pt", model, opt, scheduler, scaler, pos, identity)
        if stop_after and pos["step"] >= stop_after:
            return pos
    if not config.get("metrics_only", False):
        export_model(model, tokenizer, out / "final", {"track": config["track"], "kind": "final",
                     "step": pos["step"], "identity": identity, "template": TEMPLATE})
    write_json(out / "completion.json", {**pos, "seconds": time.monotonic() - started,
               "peak_cuda_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]})
    return pos


def run_training(config, resume=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info
    config = dict(config)
    train_records = read_json(config["train_data"])
    validation_records = read_json(config["validation_data"])
    for records, key in [(train_records, "required_train_count"), (validation_records, "required_validation_count")]:
        if key in config and len(records) != config[key]:
            raise ValueError(f"{key}={config[key]}, got {len(records)}; do not use pilot data for full runs")
    from .data import normalize
    if {normalize(r["prompt"]) for r in train_records} & {normalize(r["prompt"]) for r in validation_records}:
        raise ValueError("Training and validation prompts overlap")
    # Avoid replacing run provenance on a failed or accidental restart.
    if resume:
        saved = read_json(Path(config["output"]) / "resolved_config.json")
        if config["track"] == "sft":
            config["revision"] = saved["revision"]
    source = config["model"]
    if config["track"] != "sft":
        marker = read_json(Path(source) / "alignment_checkpoint.json")
        if marker["track"] != "sft" or marker["kind"] != "best":
            raise ValueError("Both DPO tracks must initialize policy and reference from SFT best")
        from .io import file_hash
        config["initial_checkpoint"] = {p.name: file_hash(p) for p in sorted(Path(source).glob("*.safetensors"))}
        if not config["initial_checkpoint"]:
            raise ValueError("Missing SFT checkpoint weights")
        revision = None
    else:
        revision = model_info(source, revision=config.get("revision", "main")).sha
        config["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, torch_dtype=torch.float32)
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    if config["track"] != "sft":
        reference = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float16 if config["reference_device"].startswith("cuda") else torch.float32)
        reference.to(config["reference_device"])
        reference.config.use_cache = False
    if config.get("pilot_audit"):
        import hashlib
        def state_hash(module):
            h = hashlib.sha256()
            for name, tensor in module.state_dict().items():
                h.update(name.encode())
                h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        reference_before = state_hash(reference) if reference is not None else None
    result = train(model, tokenizer, train_records, validation_records,
                   config, reference, resume)
    if config.get("pilot_audit"):
        if reference is not None:
            assert state_hash(reference) == reference_before
            assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
        reloaded = AutoModelForCausalLM.from_pretrained(Path(config["output"]) / "final", torch_dtype=torch.float32)
        model.eval()
        probe = tokenizer("Checkpoint reload probe", return_tensors="pt", return_token_type_ids=False)
        with torch.no_grad():
            expected = model(**probe.to(config["device"])).logits.float().cpu()
            actual = reloaded(**probe.to("cpu")).logits.float().cpu()
        torch.testing.assert_close(expected, actual, rtol=2e-3, atol=2e-3)
        write_json(Path(config["output"]) / "pilot_audit.json", {"checkpoint_reload": True,
                   "reference_unchanged": True if reference is not None else None,
                   "reference_hash": reference_before})
    return result
