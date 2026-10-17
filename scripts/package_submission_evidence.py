"""Copy completed, small experiment evidence into a reviewable submission tree.

This intentionally excludes large full-model weights and optimizer states. The
Kaggle kernel IDs and exact checkpoint hashes remain in the copied manifests.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.io import file_hash, read_json, read_jsonl, write_json
from alignment.pairwise import summarize


STAGES = {
    "sft": ("manifest.json", "resolved_config.json", "completion.json", "metrics.jsonl",
            "best/alignment_checkpoint.json"),
    "dpo_hh": ("manifest.json", "resolved_config.json", "completion.json", "metrics.jsonl",
               "best/alignment_checkpoint.json"),
    "dpo_ultrafeedback": ("manifest.json", "resolved_config.json", "completion.json", "metrics.jsonl",
                          "best/alignment_checkpoint.json"),
    "persona": ("manifest.json", "metrics.json", "reload_verified.json", "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors", "adapter/tokenizer.json",
                "adapter/tokenizer_config.json", "adapter/special_tokens_map.json",
                "adapter/vocab.json", "adapter/merges.txt"),
}
GENERATIONS = {"sft": ("base_alpaca.json", "sft_alpaca.json", "sft_hh.json"),
               "hh": ("dpo_hh.json",), "ultrafeedback": ("ultrafeedback_alpaca.json",),
               "persona": ("persona_base.json", "persona_lora.json")}
EVALUATIONS = {"hh": 300, "ultrafeedback": 300, "persona_style": 20, "persona_quality": 20}
DATA_FILES = ("manifest.json", "sft_manifest.json", "hh_manifest.json",
              "ultrafeedback_manifest.json", "upload_file_hashes.json")


def copy_checked(source, target, hashes, root):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if file_hash(source) != file_hash(target):
        raise IOError(f"Copy hash mismatch: {source}")
    hashes[target.relative_to(root).as_posix()] = file_hash(target)


def package(args):
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Evidence output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    stage_paths = {key: Path(getattr(args, key)) for key in STAGES}
    for key, names in STAGES.items():
        source = stage_paths[key]
        done = source / ("reload_verified.json" if key == "persona" else "completion.json")
        completion = read_json(done)
        completed = completion.get("ok") is True if key == "persona" else completion.get("step", 0) >= 1
        if not completed:
            raise ValueError(f"Stage did not complete: {source}")
        for name in names:
            copy_checked(source / name, root / "stages" / key / name, hashes, root)
    prepared = Path(args.prepared_data)
    for name in DATA_FILES:
        copy_checked(prepared / name, root / "data" / name, hashes, root)
    generation_paths = {key: Path(getattr(args, key + "_generation")) for key in GENERATIONS}
    for key, names in GENERATIONS.items():
        source = generation_paths[key]
        for name in names:
            rows = read_json(source / name)
            expected = 20 if key == "persona" else 300
            if len(rows) != expected or len({row["id"] for row in rows}) != expected:
                raise ValueError(f"Generation count/IDs mismatch: {source / name}")
            copy_checked(source / name, root / "generations" / key / name, hashes, root)
            copy_checked(source / (Path(name).stem + ".manifest.json"),
                         root / "generations" / key / (Path(name).stem + ".manifest.json"), hashes, root)
        copy_checked(source / "completion.json", root / "generations" / key / "completion.json", hashes, root)
    evaluation = Path(args.evaluation)
    for track, expected in EVALUATIONS.items():
        pairs = read_jsonl(evaluation / (track + "_pairs.jsonl"))
        judgments = read_jsonl(evaluation / (track + "_judged.jsonl"))
        attempts = read_jsonl(evaluation / (track + "_judged.attempts.jsonl"))
        actual = summarize(pairs, judgments, attempts)
        saved = read_json(evaluation / (track + "_summary.json"))
        if len(pairs) != expected or len(judgments) != expected or actual != saved:
            raise ValueError(f"Incomplete or inconsistent evaluation: {track}")
        for suffix in ("_pairs.jsonl", "_judged.jsonl", "_judged.attempts.jsonl",
                       "_judged.manifest.json", "_judged.raw.jsonl", "_summary.json"):
            name = track + suffix
            copy_checked(evaluation / name, root / "evaluation" / name, hashes, root)
        failures = evaluation / (track + "_judged.failures.jsonl")
        if failures.is_file():
            copy_checked(failures, root / "evaluation" / failures.name, hashes, root)
    write_json(root / "evidence_manifest.json", {"files_sha256": hashes,
               "excluded": "Full-model weights and optimizer state; attach Kaggle saved kernels for these."})
    print(json.dumps({"output": str(root), "files": len(hashes)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in STAGES:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in GENERATIONS:
        parser.add_argument("--" + name.replace("_", "-") + "-generation", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--prepared-data", default="kaggle_data_upload")
    parser.add_argument("--output", default="submission/evidence")
    package(parser.parse_args())
