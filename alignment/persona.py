"""The sole TRL/PEFT training track; conversational input, original-base reference."""
from pathlib import Path
from .data import select_splits, normalize
from .io import read_json, write_json, environment


def require_finite_metrics(metrics, history):
    import math
    for row in [metrics, *history]:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite persona metric {key}={value}; inspect gradients")


def conversational(row):
    return {"prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}]}


def run_persona(config, resume=None):
    import torch
    from datasets import Dataset
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, PeftModel
    from trl import DPOTrainer, DPOConfig
    out = Path(config["output"])
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Persona output already exists: {out}")
    set_seed(config["seed"])
    revision = model_info(config["model"], revision=config.get("revision", "main")).sha
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision, torch_dtype=torch.float32)
    targets = config["target_modules"]
    names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if not set(targets) <= names:
        raise ValueError(f"Missing LoRA target modules: {set(targets) - names}")
    raw = read_json(config["data"])
    excluded = [r["instruction"] for r in read_json(config["evaluation_prompts"])]
    if any(normalize(r["prompt"]) in {normalize(p) for p in excluded} for r in raw):
        raise ValueError("Persona evaluation overlaps training data")
    train, val, counts = select_splits(raw, "persona", len(raw) - config["validation_size"],
                                     config["validation_size"], config["seed"], excluded)
    lora = LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                      target_modules=targets, bias="none", task_type="CAUSAL_LM")
    args = DPOConfig(output_dir=str(out), learning_rate=config["learning_rate"], beta=config["beta"],
                     rpo_alpha=config.get("rpo_alpha"),
                     per_device_train_batch_size=config["microbatch"], per_device_eval_batch_size=1,
                     gradient_accumulation_steps=config["accumulation"], num_train_epochs=config["epochs"],
                     max_steps=config.get("max_updates", -1), max_length=config["max_length"],
                     max_prompt_length=config.get("max_prompt_length", 256), fp16=False, bf16=False,
                     optim="adamw_torch", weight_decay=config.get("weight_decay", 0.0),
                     lr_scheduler_type=config.get("scheduler", "constant"),
                     warmup_ratio=config.get("warmup_ratio", 0.0),
                     max_grad_norm=1.0, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                     logging_steps=1, report_to="none", seed=config["seed"], data_seed=config["seed"],
                     dataloader_pin_memory=torch.cuda.is_available(),
                     remove_unused_columns=False, gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = DPOTrainer(model=model, ref_model=None, args=args, processing_class=tokenizer,
                          peft_config=lora, train_dataset=Dataset.from_list([conversational(r) for r in train]),
                          eval_dataset=Dataset.from_list([conversational(r) for r in val]))
    if not isinstance(trainer.model, PeftModel) or trainer.ref_model is not None:
        raise RuntimeError("Expected PEFT adapter-disabled original-base reference")
    # TRL v0.18.1 null_ref_context disables the adapter when ref_model=None.
    with trainer.null_ref_context():
        if not all(m.disable_adapters for m in trainer.model.modules() if hasattr(m, "disable_adapters")):
            raise RuntimeError("Reference context did not disable adapters")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    write_json(out / "manifest.json", {**environment(), "config": {**config, "revision": revision},
               "trainable": trainable, "total": total, "fraction": trainable / total,
               "counts": counts, "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in val],
               "reference": "original base via disabled LoRA adapters"})
    before = {name: p.detach().cpu().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    require_finite_metrics(metrics, trainer.state.log_history)
    if not all(torch.isfinite(p).all() for p in trainer.model.parameters() if p.requires_grad):
        raise FloatingPointError("Nonfinite trained LoRA adapter weights")
    if not any(not torch.equal(before[name], p.detach().cpu())
               for name, p in trainer.model.named_parameters() if p.requires_grad):
        raise RuntimeError("Persona LoRA parameters did not change after training")
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(out / "adapter")
    trainer.save_state()
    write_json(out / "metrics.json", {"evaluation": metrics, "history": trainer.state.log_history})
    # Verify the saved adapter can actually reload onto the pinned original base.
    base = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision, torch_dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, out / "adapter").eval()
    trainer.model.eval()
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": excluded[0]}],
                                           add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        expected = trainer.model(input_ids=inputs.to(trainer.model.device)).logits[:, -1].float().cpu()
        actual = reloaded(input_ids=inputs).logits[:, -1].float().cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    write_json(out / "reload_verified.json", {"ok": True, "base_revision": revision})
