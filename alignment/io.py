"""Small atomic artifacts, identities, and append-only journals."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def environment():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "kaggle-benchmarks"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "alignment").glob("*.py")) + [root / "submission_adapter.py"]
    source_hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in source_files if p.exists()}
    return {"commit": commit, "versions": versions, "source_hash": digest(source_hashes),
            "source_files": source_hashes}


def load_config(path):
    source = Path(path).read_text(encoding="utf-8")
    # The committed configs use JSON syntax (also accepted by YAML). PyYAML
    # treats exponent-form JSON numbers such as 2e-05 as strings.
    try:
        return json.loads(source)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(source)
