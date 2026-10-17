"""Select deterministic judged W/L/T examples whose displayed texts support the label."""
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def excerpt(text, limit=300):
    text = " ".join(str(text).split()).replace(chr(0xFFFD), "'")
    if len(text) <= limit:
        return text
    window = text[:limit]
    boundaries = [window.rfind(mark) for mark in (". ", "! ", "? ")]
    boundary = max(boundaries)
    if boundary >= limit // 2:
        return window[:boundary + 1] + " …"
    return window.rsplit(" ", 1)[0] + " …"


def select_files(track, pair_path, judged_path, candidate_label="DPO", baseline_label="SFT"):
    pairs = {row["id"]: row for row in load_jsonl(pair_path)}
    found = {}
    for judged in load_jsonl(judged_path):
        pair = pairs[judged["id"]]
        decision = judged["decision"]
        outcome = "tie" if decision == "tie" else ("win" if pair["mapping"][decision] == pair["candidate"] else "loss")
        candidate_side = next(side for side, model in pair["mapping"].items() if model == pair["candidate"])
        baseline_side = "B" if candidate_side == "A" else "A"
        limit = 360 if track == "hh" and outcome == "win" else 300
        candidate, baseline = excerpt(pair[candidate_side], limit), excerpt(pair[baseline_side], limit)
        if outcome in found or (outcome != "tie" and candidate == baseline):
            continue
        found[outcome] = {"id": pair["id"], "prompt_id": pair["prompt_id"],
                          "prompt": excerpt(pair["instruction"]),
                          "candidate": candidate, "baseline": baseline,
                          "candidate_label": candidate_label, "baseline_label": baseline_label,
                          "analysis": (f"The blinded judge preferred the {candidate_label} response; the displayed texts differ."
                                       if outcome == "win" else
                                       f"The blinded judge preferred the {baseline_label} response; the displayed texts differ."
                                       if outcome == "loss" else
                                       "The blinded judge found no material advantage; identical text is allowed for ties.")}
    if set(found) != {"win", "loss", "tie"}:
        raise ValueError(f"{track}: could not find valid win, loss, and tie examples")
    return found


def select(track):
    base = Path("submission/evidence/evaluation")
    return select_files(track, base / f"{track}_pairs.jsonl", base / f"{track}_judged.jsonl")


def main():
    result = {track: select(track) for track in ("hh", "ultrafeedback")}
    judge_roots = list(Path("outputs/persona_v3_judging").rglob("judging"))
    persona_pairs = Path("outputs/persona_v3_pairs/persona_v3b_style_pairs.jsonl")
    if len(judge_roots) == 1 and persona_pairs.exists():
        result["persona_v3b"] = select_files("persona_v3b_style", persona_pairs,
            judge_roots[0] / "persona_v3b_style_judged.jsonl", "v3b adapter", "base")
    out = Path("outputs/analysis/report_examples.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: list(v) for k, v in result.items()}))


if __name__ == "__main__":
    main()
