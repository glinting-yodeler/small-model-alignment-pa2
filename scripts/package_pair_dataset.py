"""Verify and stage immutable blinded-pair inputs for a private Kaggle judge."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from alignment.io import digest, read_jsonl, write_json


def package(paths, output, dataset_id, expected_counts):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Pair upload already exists: {output}")
    output.mkdir(parents=True)
    hashes = {}
    for path, count in zip(paths, expected_counts):
        path = Path(path)
        pairs = read_jsonl(path)
        if len(pairs) != count or len({p["id"] for p in pairs}) != count:
            raise ValueError(f"Unexpected pair count or duplicate IDs in {path}")
        for pair in pairs:
            if pair["hash"] != digest({k: v for k, v in pair.items() if k != "hash"}):
                raise ValueError(f"Pair hash mismatch in {path}")
        target = output / path.name
        shutil.copy2(path, target)
        hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(output / "pair_file_hashes.json", hashes)
    write_json(output / "dataset-metadata.json", {"title": dataset_id.rsplit("/", 1)[1].replace("-", " "),
               "id": dataset_id, "licenses": [{"name": "other"}]})
    print(f"Verified {len(paths)} pair files at {output}; upload privately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--count", type=int, action="append", required=True)
    args = parser.parse_args()
    if len(args.pair) != len(args.count):
        raise SystemExit("One --count per --pair is required")
    package(args.pair, args.output, args.dataset_id, args.count)
