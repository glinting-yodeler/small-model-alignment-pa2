"""Record the assignment GPU and library environment without changing it."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess


PACKAGES = ("torch", "transformers", "datasets", "trl", "peft", "accelerate")


def collect():
    import torch
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False)
        nvidia = query.stdout.strip() or query.stderr.strip()
    except FileNotFoundError:
        nvidia = "nvidia-smi unavailable"
    count = torch.cuda.device_count()
    return {"nvidia_smi": nvidia, "cuda_device_count": count,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(count)],
            "versions": {name: importlib.metadata.version(name) for name in PACKAGES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/analysis/environment.json")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    result = collect()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["cuda_device_count"] < 2 and not args.allow_cpu:
        raise SystemExit("ERROR: fewer than two GPUs are visible; select Kaggle GPU T4 x2 or pass --allow-cpu.")


if __name__ == "__main__":
    main()
