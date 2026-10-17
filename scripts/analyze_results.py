"""Quantitative post-evaluation analysis from saved generations and judgments."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def words(text):
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__); result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        value = (i + j - 1) / 2 + 1
        for k in order[i:j]: result[k] = value
        i = j
    return result


def correlation(a, b):
    ra, rb = rank(a), rank(b); ma, mb = statistics.mean(ra), statistics.mean(rb)
    numerator = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    denominator = math.sqrt(sum((x-ma)**2 for x in ra) * sum((y-mb)**2 for y in rb))
    return numerator / denominator if denominator else None


def output_stats(rows, tokenizer):
    wc = [len(words(r["output"])) for r in rows]
    tc = [len(tokenizer.encode(r["output"], add_special_tokens=False)) for r in rows]
    degenerate = [len(w) > 40 and len({x.lower() for x in w}) / len(w) < .3
                  for w in map(lambda r: words(r["output"]), rows)]
    return {"n": len(rows), "mean_words": statistics.mean(wc), "median_words": statistics.median(wc),
            "mean_tokens": statistics.mean(tc), "median_tokens": statistics.median(tc),
            "retokenized_decoded_length_ge_256_pct": 100 * sum(n >= 256 for n in tc) / len(tc),
            "degeneration_pct": 100 * sum(degenerate) / len(rows)}


def mapped_outcome(pair, judgment):
    decision = judgment["decision"]
    if decision in ("tie", "unknown"): return decision
    return "win" if pair["mapping"][decision] == pair["candidate"] else "loss"


def bootstrap_ci(outcomes, seed=42, samples=10000):
    values = [1 if x == "win" else 0 if x == "loss" else .5 for x in outcomes]
    rng = random.Random(seed); n = len(values)
    scores = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return [scores[int(.025*samples)], scores[int(.975*samples)-1]]


def evaluate(track, root):
    pairs = load_jsonl(root / f"{track}_pairs.jsonl")
    judgments = {x["id"]: x for x in load_jsonl(root / f"{track}_judged.jsonl")}
    outcomes = [mapped_outcome(p, judgments[p["id"]]) for p in pairs]
    return pairs, judgments, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default="outputs/evaluation")
    parser.add_argument("--output", default="outputs/analysis/analysis.json")
    args = parser.parse_args(); evaluation = Path(args.evaluation)
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987", local_files_only=True)
    persona = AutoTokenizer.from_pretrained("submission/evidence/stages/persona/adapter", local_files_only=True)
    sources = {
        "alpaca_base": load_json("outputs/generation-sft-v1/outputs/generations/base_alpaca.json"),
        "alpaca_sft": load_json("outputs/generation-sft-v1/outputs/generations/sft_alpaca.json"),
        "alpaca_uf": load_json("outputs/generation-uf-v1/outputs/generations/ultrafeedback_alpaca.json"),
        "hh_sft": load_json("outputs/generation-sft-v1/outputs/generations/sft_hh.json"),
        "hh_dpo": load_json("outputs/generation-hh-v1/outputs/generations/dpo_hh.json"),
        "persona_base": load_json("outputs/persona-generation-v1/outputs/generations/persona_base.json"),
        "persona_lora": load_json("outputs/persona-generation-v1/outputs/generations/persona_lora.json")}
    stats = {name: output_stats(rows, persona if name.startswith("persona") else qwen)
             for name, rows in sources.items()}
    for left, right in (("alpaca_sft", "alpaca_uf"), ("hh_sft", "hh_dpo"),
                        ("persona_base", "persona_lora")):
        identical = sum(a["output"] == b["output"] for a, b in zip(sources[left], sources[right]))
        stats[left]["identical_to_comparison_pct"] = 100 * identical / len(sources[left])
        stats[right]["identical_to_comparison_pct"] = 100 * identical / len(sources[right])

    evaluations = {}; cis = {}
    for track in ("hh", "ultrafeedback", "persona_style", "persona_quality"):
        pairs, judgments, outcomes = evaluate(track, evaluation)
        counts = Counter(outcomes); n = len(outcomes)
        evaluations[track] = {**counts, "n": n, "tie_adjusted":
                              (counts["win"] + .5*counts["tie"]) / n}
        cis[track] = bootstrap_ci(outcomes)

    pairs, judgments, outcomes = evaluate("ultrafeedback", evaluation)
    sft = {r["id"]: r for r in sources["alpaca_sft"]}; dpo = {r["id"]: r for r in sources["alpaca_uf"]}
    deltas = [len(qwen.encode(dpo[p["prompt_id"]]["output"], add_special_tokens=False)) -
              len(qwen.encode(sft[p["prompt_id"]]["output"], add_special_tokens=False)) for p in pairs]
    ordered = sorted(range(len(pairs)), key=deltas.__getitem__); buckets = {}
    for label, ids in zip(("shorter", "middle", "longer"), (ordered[:100], ordered[100:200], ordered[200:])):
        c = Counter(outcomes[i] for i in ids); buckets[label] = {**c, "n": len(ids),
            "delta_min": min(deltas[i] for i in ids), "delta_max": max(deltas[i] for i in ids)}
    score = [1 if x == "win" else -1 if x == "loss" else 0 for x in outcomes]
    uf = {"token_length_delta_terciles": buckets, "spearman_delta_vs_outcome": correlation(deltas, score),
          "mean_token_delta": statistics.mean(deltas), "mean_token_delta_pct_vs_sft":
          100 * statistics.mean(deltas) / stats["alpaca_sft"]["mean_tokens"]}

    pairs, judgments, outcomes = evaluate("hh", evaluation)
    meta = {r["id"]: r for r in load_json("eval_sets/hh_rlhf_eval_300_seed42.json")}
    breakdown = {"category": defaultdict(Counter), "expected_behavior": defaultdict(Counter)}
    for pair, outcome in zip(pairs, outcomes):
        row = meta[pair["prompt_id"]]
        for field in breakdown: breakdown[field][row[field]][outcome] += 1
    seen = set(); unique_outcomes = []
    for pair, outcome in zip(pairs, outcomes):
        key = " ".join(pair["instruction"].lower().split())
        if key not in seen: seen.add(key); unique_outcomes.append(outcome)
    uc = Counter(unique_outcomes)
    hh_sft = {r["id"]: r for r in sources["hh_sft"]}; hh_dpo = {r["id"]: r for r in sources["hh_dpo"]}
    hh_deltas = [len(qwen.encode(hh_dpo[p["prompt_id"]]["output"], add_special_tokens=False)) -
                 len(qwen.encode(hh_sft[p["prompt_id"]]["output"], add_special_tokens=False)) for p in pairs]
    hh_order = sorted(range(len(pairs)), key=hh_deltas.__getitem__); hh_buckets = {}
    for label, ids in zip(("shorter", "middle", "longer"),
                          (hh_order[:100], hh_order[100:200], hh_order[200:])):
        c = Counter(outcomes[i] for i in ids); hh_buckets[label] = {**c, "n": len(ids),
            "delta_min": min(hh_deltas[i] for i in ids), "delta_max": max(hh_deltas[i] for i in ids)}
    def degenerate_text(text):
        tokens = words(text)
        return len(tokens) > 40 and len({x.lower() for x in tokens}) / len(tokens) < .3
    deg_groups = defaultdict(Counter)
    for pair, outcome in zip(pairs, outcomes):
        a = degenerate_text(hh_sft[pair["prompt_id"]]["output"])
        b = degenerate_text(hh_dpo[pair["prompt_id"]]["output"])
        label = "both repetitive" if a and b else "only SFT repetitive" if a else "only DPO repetitive" if b else "neither repetitive"
        deg_groups[label][outcome] += 1
    hh_score = [1 if x == "win" else -1 if x == "loss" else 0 for x in outcomes]
    hh = {"by_category": {k: dict(v) for k,v in breakdown["category"].items()},
          "by_expected_behavior": {k: dict(v) for k,v in breakdown["expected_behavior"].items()},
          "unique_prompt_rule": "Keep the first saved row for each whitespace-normalized, lowercased prompt text.",
          "unique_prompts": {**uc, "n": len(unique_outcomes), "tie_adjusted":
                             (uc["win"]+.5*uc["tie"])/len(unique_outcomes),
                             "bootstrap_95pct_ci": bootstrap_ci(unique_outcomes)},
          "token_length_delta_terciles": hh_buckets,
          "spearman_delta_vs_outcome": correlation(hh_deltas, hh_score),
          "outcome_by_degeneration": {k: dict(v) | {"n": sum(v.values())} for k,v in deg_groups.items()}}
    result = {"definitions": {"degeneration": "output has >40 words and unique-word ratio <0.3",
              "retokenized_decoded_length_ge_256":
              "encode saved decoded output text with the pinned tokenizer and test length >=256; "
              "this is a proxy, not a confirmed generation-limit hit, because special-token removal, "
              "decoding, and re-tokenization can change length",
              "bootstrap": "10,000 row-level resamples, seed 42"},
              "length_and_repetition": stats, "evaluations": evaluations,
              "bootstrap_95pct_ci": cis, "ultrafeedback": uf, "hh": hh}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Base", "SFT", "UF DPO"]; names = ["alpaca_base", "alpaca_sft", "alpaca_uf"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].bar(labels, [stats[n]["mean_tokens"] for n in names], color=["#718096","#2B6CB0","#C05621"])
    axes[0].set(ylabel="Mean output tokens", title="AlpacaEval response length")
    axes[1].bar(labels, [stats[n]["degeneration_pct"] for n in names], color=["#718096","#2B6CB0","#C05621"])
    axes[1].set(ylabel="Degeneration (%)", title="Repetition metric")
    fig.tight_layout(); fig.savefig(output.parent/"length_repetition.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 2.7)); x=list(buckets)
    bottom=[0]*3
    for outcome,color in (("win","#2F855A"),("loss","#C53030"),("tie","#718096")):
        vals=[100*buckets[k][outcome]/buckets[k]["n"] for k in x]
        ax.bar(x,vals,bottom=bottom,label=outcome,color=color);bottom=[a+b for a,b in zip(bottom,vals)]
    ax.set(ylabel="Rows (%)", xlabel="DPO minus SFT token-length tercile", title="UltraFeedback outcome by length change")
    ax.legend(ncol=3); fig.tight_layout(); fig.savefig(output.parent/"uf_length_outcomes.png",dpi=180);plt.close(fig)
    print(json.dumps({"output": str(output), "figures": 2}, indent=2))


if __name__ == "__main__": main()
