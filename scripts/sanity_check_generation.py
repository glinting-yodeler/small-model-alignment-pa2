"""Inspect 2–5 deterministic generations before launching a full fixed set."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.formatting import TEMPLATE, format_prompt
from alignment.io import load_config, read_json


def inspect_rows(config, tokenizer, model, prompts):
    import torch
    records = []
    for row in prompts:
        formatted = (tokenizer.apply_chat_template(
            [{"role": "user", "content": row["instruction"]}], tokenize=False,
            add_generation_prompt=True) if config.get("format") == "chat"
            else format_prompt(row["instruction"]))
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt",
                           return_token_type_ids=False)
        inputs = inputs.to(config["device"]) if hasattr(inputs, "to") else inputs
        with torch.inference_mode():
            # BatchEncoding exposes internal attributes through vars(); only
            # pass the actual tensor mapping accepted by generate().
            model_inputs = dict(inputs)
            generated = model.generate(
                **model_inputs,
                do_sample=False, max_new_tokens=config["max_new_tokens"],
                repetition_penalty=config.get("repetition_penalty", 1.0),
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prompt_width = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else inputs["input_ids"].shape[1]
        continuation = generated[0, prompt_width:].detach().cpu().tolist()
        records.append({"id": row["id"], "checkpoint": config["model"],
                        "adapter": config.get("adapter"),
                        "tokenizer": config["model"], "formatted_prompt": formatted,
                        "raw_output": tokenizer.decode(continuation, skip_special_tokens=False),
                        "eos_present": tokenizer.eos_token_id in continuation,
                        "token_count": len(continuation), "template":
                        tokenizer.chat_template if config.get("format") == "chat" else TEMPLATE})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=3, choices=range(2, 6))
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision"))
    if config.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config["adapter"])
    model.to(config["device"]).eval()
    rows = inspect_rows(config, tokenizer, model, read_json(config["prompts"])[:args.count])
    name = Path(config["output"]).stem
    output = Path(args.output or f"outputs/analysis/sanity_{name}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
