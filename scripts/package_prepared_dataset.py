"""Flatten and verify frozen subsets for a private Kaggle input dataset."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.io import digest, read_json, file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/prepared"
    out = root / "kaggle_data_upload"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace prepared upload: {out}")
    master = read_json(source / "manifest.json")
    expected = {"sft": (30000, 1000), "hh": (15000, 1000), "ultrafeedback": (15000, 1000)}
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for track, (n_train, n_val) in expected.items():
        manifest = read_json(source / track / "manifest.json")
        train = read_json(source / track / "train.json")
        val = read_json(source / track / "validation.json")
        if (len(train), len(val)) != (n_train, n_val):
            raise ValueError(f"Wrong {track} counts: {len(train)}, {len(val)}")
        if digest(train) != manifest["train_hash"] or digest(val) != manifest["validation_hash"]:
            raise ValueError(f"{track} subset differs from frozen manifest")
        for split, path in [("train", source / track / "train.json"),
                            ("validation", source / track / "validation.json"),
                            ("manifest", source / track / "manifest.json")]:
            name = f"{track}_{split}.json"
            shutil.copyfile(path, out / name)
            files[name] = file_hash(out / name)
    alpaca = read_json(source / "alpaca_eval_300_seed42.json")
    if len(alpaca) != 300 or digest(alpaca) != master["alpaca_hash"]:
        raise ValueError("Fixed Alpaca prompt manifest differs")
    for name in ("alpaca_eval_300_seed42.json", "manifest.json"):
        shutil.copyfile(source / name, out / name)
        files[name] = file_hash(out / name)
    (out / "upload_file_hashes.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "PA2 Frozen Prepared Inputs October 2026",
        "id": "moibbi/pa2-frozen-prepared-inputs",
        "licenses": [{"name": "other"}],
        "description": "Private course experiment inputs. Original sources: UltraChat 200k, Anthropic HH-RLHF, UltraFeedback binarized (MIT); AlpacaEval (CC-BY-NC-4.0). See source revisions and licenses in manifest.json."
    }, indent=2), encoding="utf-8")
    print(f"Verified {len(files)} files at {out}; CLI defaults to private upload.")


if __name__ == "__main__":
    main()
