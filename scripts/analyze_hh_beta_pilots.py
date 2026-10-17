"""Extract matched HH beta validation curves from downloaded Kaggle metrics."""
import json
from pathlib import Path


def main():
    root = Path("outputs/hh_beta_pilots_v2")
    runs = []
    for path in sorted(root.rglob("metrics.jsonl")):
        points = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        validation = [row for row in points if row.get("phase") == "validation" and row.get("step") in (50, 100, 150)]
        if not validation:
            continue
        config_path = path.with_name("resolved_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        runs.append({"beta": config["beta"], "path": path.as_posix(),
                     "validation": [{"step": row["step"], "reward_accuracy": row["reward_accuracy"],
                                     "reward_margin": row["reward_margin"], "loss": row["loss"]}
                                    for row in validation]})
    runs.sort(key=lambda row: row["beta"])
    if len(runs) != 3 or any([p["step"] for p in run["validation"]] != [50, 100, 150] for run in runs):
        raise ValueError(f"Expected three complete beta curves, found {[(r['beta'], len(r['validation'])) for r in runs]}")
    out = Path("outputs/analysis/hh_beta_pilots.json")
    out.write_text(json.dumps({"purpose": "validation diagnostics only; submitted HH checkpoint unchanged",
                               "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "betas": [r["beta"] for r in runs]}))


if __name__ == "__main__":
    main()
