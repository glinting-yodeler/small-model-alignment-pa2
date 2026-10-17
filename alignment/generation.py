"""Greedy response-only generation with checkpoint/config-bound resume."""
from pathlib import Path
import torch
from .io import digest, file_hash, read_json, write_json, read_jsonl, append_jsonl, environment
from .formatting import format_prompt, TEMPLATE


def generate(config):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import model_info
    config = dict(config)
    model_path = Path(config["model"])
    if model_path.is_dir():
        checkpoint = {p.name: file_hash(p) for p in sorted(model_path.glob("*.safetensors"))}
        if not checkpoint:
            raise ValueError("No checkpoint weights found")
        revision = None
    else:
        revision = model_info(config["model"], revision=config.get("revision", "main")).sha
        checkpoint = revision
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=revision,
                torch_dtype=torch.float16 if config["device"].startswith("cuda") else torch.float32)
    if config.get("adapter"):
        from peft import PeftModel
        checkpoint = {"base": checkpoint, "adapter": {p.name: file_hash(p) for p in Path(config["adapter"]).glob("*.safetensors")}}
        model = PeftModel.from_pretrained(model, config["adapter"])
    model.to(config["device"]).eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = read_json(config["prompts"])
    if len({r["id"] for r in prompts}) != len(prompts):
        raise ValueError("Duplicate generation IDs")
    identity = digest({"config": config, "checkpoint": checkpoint, "prompts": prompts,
                       "template": tokenizer.chat_template if config.get("format") == "chat" else TEMPLATE,
                       "source_hash": environment()["source_hash"]})
    out = Path(config["output"])
    manifest = out.with_suffix(".manifest.json")
    journal = out.with_suffix(".jsonl")
    if manifest.exists():
        if read_json(manifest)["identity"] != identity:
            raise ValueError("Generation resume identity mismatch")
    elif journal.exists() or out.exists():
        raise ValueError("Existing generations lack an identity manifest")
    write_json(manifest, {**environment(), "identity": identity, "config": config, "checkpoint": checkpoint})
    prior = read_jsonl(journal)
    done = {r["id"]: r for r in prior}
    if len(done) != len(prior) or not set(done) <= {r["id"] for r in prompts}:
        raise ValueError("Invalid generation journal IDs")
    pending = [r for r in prompts if r["id"] not in done]
    for start in range(0, len(pending), config["batch_size"]):
        rows = pending[start:start + config["batch_size"]]
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["instruction"]}],
                  tokenize=False, add_generation_prompt=True) if config.get("format") == "chat"
                 else format_prompt(r["instruction"]) for r in rows]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt",
                           return_token_type_ids=False)
        if inputs.input_ids.shape[1] + config["max_new_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Prompt plus generation exceeds context; no silent truncation")
        inputs = inputs.to(config["device"])
        with torch.inference_mode():
            tokens = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
                     repetition_penalty=config.get("repetition_penalty", 1.0),
                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for row, text in zip(rows, outputs):
            value = {**row, "output": text, "generator": config["generator"]}
            append_jsonl(journal, value)
            done[row["id"]] = value
    write_json(out, [done[r["id"]] for r in prompts])
