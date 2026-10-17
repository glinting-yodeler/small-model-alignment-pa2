"""Validate and summarize the generation-level HH beta comparison."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.analyze_results import bootstrap_ci, mapped_outcome


ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("hh_beta005_vs_beta01", "hh_beta03_vs_beta01")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def unique(pattern):
    matches = list(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {pattern}, found {matches}")
    return matches[0]


def main():
    generation_root = ROOT / "outputs/hh_beta_generation/outputs/generations"
    kernel_summary = json.loads((ROOT / "outputs/hh_beta_generation/outputs/beta_generation_summary.json").read_text())
    generation = {}
    for run in kernel_summary["runs"]:
        path = generation_root / f"{run['label']}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(rows) != 300 or len({row["id"] for row in rows}) != 300 or any(not row["output"].strip() for row in rows):
            raise ValueError(f"Invalid generation coverage for {run['label']}")
        if digest != run["generation_sha256"]:
            raise ValueError(f"Generation hash mismatch for {run['label']}")
        generation[run["label"]] = {"count": 300, "sha256": digest}
    comparisons = {}
    for track in TRACKS:
        pairs_path = ROOT / f"outputs/hh_beta_pairs/{track}_pairs.jsonl"
        judged_path = unique(f"outputs/hh_beta_judging/**/judging/{track}_judged.jsonl")
        summary_path = unique(f"outputs/hh_beta_judging/**/judging/{track}_summary.json")
        pairs = load_jsonl(pairs_path); judged = load_jsonl(judged_path)
        decisions = {row["id"]: row for row in judged}
        if len(pairs) != 300 or len(decisions) != 300 or set(decisions) != {row["id"] for row in pairs}:
            raise ValueError(f"Incomplete pair/judgment coverage for {track}")
        outcomes = [mapped_outcome(pair, decisions[pair["id"]]) for pair in pairs]
        counts = Counter(outcomes); score = (counts["win"] + .5 * counts["tie"]) / 300
        saved = json.loads(summary_path.read_text(encoding="utf-8"))[track]
        if (counts["win"], counts["loss"], counts["tie"]) != (saved["W"], saved["L"], saved["T"]):
            raise ValueError(f"Saved summary mismatch for {track}")
        comparisons[track] = {
            "n": 300, "win": counts["win"], "loss": counts["loss"], "tie": counts["tie"],
            "unknown": saved["U"], "failure": saved["F"], "tie_adjusted": score,
            "bootstrap_95pct_ci": bootstrap_ci(outcomes), "judge": "google/gemini-3.6-flash",
            "runner": "gemini-3.8-flash", "pair_retries_including_failures": saved["pair_retries_including_failures"],
            "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
            "judgments_sha256": hashlib.sha256(judged_path.read_bytes()).hexdigest()}
    result = {"generation": generation, "comparisons": comparisons,
              "bootstrap": "10,000 row-level resamples, seed 42"}
    out = ROOT / "outputs/analysis/hh_beta_generation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
