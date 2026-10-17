"""Apply the predeclared persona-v2 selection rule to held-out generations."""
import argparse
import json
from pathlib import Path


def words(text):
    return str(text).lower().split()


def generation_metrics(base_rows, rows, markers):
    base = {row["id"]: row["output"] for row in base_rows}
    current = {row["id"]: row["output"] for row in rows}
    if set(current) != set(base) or not current:
        raise ValueError("Every model must contain the same nonempty held-out prompt IDs")
    ordered = sorted(base)
    lengths = [len(words(current[key])) for key in ordered]
    degeneration = [n > 40 and len(set(words(current[key]))) / n < 0.3
                    for key, n in zip(ordered, lengths)]
    changed = [current[key].strip() != base[key].strip() for key in ordered]
    marked = [bool(set(words(current[key])) & set(markers)) for key in ordered]
    n = len(ordered)
    return {"n": n, "degeneration_rate": sum(degeneration) / n,
            "changed_vs_base_rate": sum(changed) / n,
            "style_marker_rate": sum(marked) / n, "mean_words": sum(lengths) / n}


def choose(rows, base_degeneration_rate):
    for row in rows:
        row["eligible"] = (row["degeneration_rate"] <= base_degeneration_rate + 0.05
                           and row["changed_vs_base_rate"] >= 0.30)
    eligible = [row for row in rows if row["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda row: (row["style_marker_rate"],
                       row["held_out_reward_accuracy"], -row["held_out_eval_loss"], row["name"]))
        reason = "highest style-marker rate among eligible adapters; reward accuracy and eval loss are tie-breakers"
    else:
        selected = min(rows, key=lambda row: (row["degeneration_rate"], row["name"]))
        reason = "no adapter met eligibility; selected the lowest degeneration rate"
    return selected["name"], reason


def run(catalog_path, output_path):
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    base_rows = json.loads(Path(catalog["base_generation"]).read_text(encoding="utf-8"))
    markers = catalog["style_markers"]
    base = generation_metrics(base_rows, base_rows, markers)
    rows = []
    for item in catalog["adapters"]:
        metrics = json.loads(Path(item["metrics"]).read_text(encoding="utf-8"))["evaluation"]
        row = {"name": item["name"], **generation_metrics(base_rows,
               json.loads(Path(item["generation"]).read_text(encoding="utf-8")), markers),
               "held_out_reward_accuracy": metrics["eval_rewards/accuracies"],
               "held_out_reward_margin": metrics["eval_rewards/margins"],
               "held_out_eval_loss": metrics["eval_loss"]}
        rows.append(row)
    selected, reason = choose(rows, base["degeneration_rate"])
    result = {"rule": "docs/persona_selection_rule_v2.md", "base": base,
              "adapters": rows, "selected": selected, "reason": reason}
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", default="outputs/analysis/persona_selection_v2.json")
    args = parser.parse_args()
    print(json.dumps(run(args.catalog, args.output), indent=2))
