"""Download only in an explicitly invoked preparation command; freeze revisions."""
from pathlib import Path
import random
from .data import select_splits, save_splits
from .io import read_json, write_json, digest
from .training import encode_records


def prepare(config, pilot=False):
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    api = HfApi()
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared output already exists: {output}")
    sources = config["sources"]
    resolved = {}
    def dataset(key):
        spec = sources[key]
        info = api.dataset_info(spec["repo"], revision=spec.get("revision", "main"))
        resolved[key] = {**spec, "revision": info.sha,
                         "license": getattr(info.card_data, "license", None)}
        if spec.get("json_file"):
            return read_json(hf_hub_download(spec["repo"], spec["json_file"], repo_type="dataset", revision=info.sha))
        return load_dataset(spec["repo"], name=spec.get("name"), split=spec["split"],
                            data_files=spec.get("data_files"), revision=info.sha, streaming=key != "alpaca")
    alpaca = list(dataset("alpaca"))
    indices = random.Random(42).sample(range(len(alpaca)), 300)
    evaluation = [{"id": f"alpaca-{i}", "instruction": alpaca[i]["instruction"], "dataset": "alpaca_eval"}
                  for i in indices]
    write_json(output / "alpaca_eval_300_seed42.json", evaluation)
    excluded = [r["instruction"] for r in evaluation]
    for path in config["exclude_sets"]:
        excluded.extend(r["instruction"] for r in read_json(path))
    model_revision = api.model_info(config["tokenizer"], revision=config.get("tokenizer_revision", "main")).sha
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], revision=model_revision)
    for track in ("sft", "hh", "ultrafeedback"):
        rows = dataset(track)
        if pilot:
            import itertools
            rows = itertools.islice(rows, config.get("pilot_scan_rows", 5000))
        cfg = {"track": "sft" if track == "sft" else "dpo", "max_length": 512,
               **({"max_prompt_length": 256} if track != "sft" else {})}
        def scoreable(row):
            try:
                encode_records(tokenizer, [row], cfg)
                return True
            except ValueError:
                return False
        n, v = (16, 4) if pilot else (config["counts"][track], 1000)
        train, val, counts = select_splits(rows, track, n, v, config["seed"], excluded,
                                         source=sources[track]["repo"], validator=scoreable)
        save_splits(output / track, train, val, {"source": resolved[track], "seed": config["seed"],
                     "counts": counts, "tokenizer_revision": model_revision, "encoding": cfg,
                     "excluded_hash": digest(excluded), "pilot": pilot})
    write_json(output / "manifest.json", {"sources": resolved, "seed": config["seed"],
               "alpaca_source_indices": indices, "alpaca_hash": digest(evaluation), "pilot": pilot,
               "tokenizer_revision": model_revision})
