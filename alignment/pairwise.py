"""ID joins, fixed A/B randomization, private mappings, and explicit denominators."""
from collections import Counter
import random
from .io import digest


def index_rows(rows):
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate IDs")
    return result


def build_pairs(baseline, candidate, track, seed=42, expected_count=300):
    left, right = index_rows(baseline), index_rows(candidate)
    if left.keys() != right.keys() or len(left) != expected_count:
        raise ValueError("Generation ID sets/count do not match")
    rng, pairs = random.Random(seed), []
    for key, a in left.items():
        b = right[key]
        if a["instruction"] != b["instruction"] or a["dataset"] != b["dataset"]:
            raise ValueError("Generation instructions/datasets differ")
        if a["generator"] == b["generator"]:
            raise ValueError("Comparisons require distinct generator identities")
        ordered = [a, b]
        rng.shuffle(ordered)
        pair = {"id": digest([track, key])[:24], "prompt_id": key, "track": track,
                "instruction": a["instruction"], "A": ordered[0]["output"], "B": ordered[1]["output"],
                "mapping": {"A": ordered[0]["generator"], "B": ordered[1]["generator"]},
                "candidate": b["generator"], "order_seed": seed}
        pair["hash"] = digest(pair)
        pairs.append(pair)
    return pairs


def summarize(pairs, judgments, attempts=()):
    index_rows(pairs)
    results = index_rows(judgments)
    if not set(results) <= {p["id"] for p in pairs}:
        raise ValueError("Judgments include unexpected IDs")
    by_track = {}
    for pair in pairs:
        counts = by_track.setdefault(pair["track"], Counter(N=0, W=0, L=0, T=0, U=0, F=0, retries=0))
        counts["N"] += 1
        row = results.get(pair["id"])
        if row is None:
            counts["F"] += 1
            continue
        if row["pair_hash"] != pair["hash"]:
            raise ValueError("Judgment content identity differs")
        winner = row["decision"]
        if winner not in ("A", "B", "tie", "unknown"):
            raise ValueError("Invalid decision")
        outcome = "T" if winner == "tie" else "U" if winner == "unknown" else (
            "W" if pair["mapping"][winner] == pair["candidate"] else "L")
        counts[outcome] += 1
        counts["retries"] += max(0, row["attempt"] - 1)
    result = {}
    for track, c in by_track.items():
        ratio = lambda n, d: n / d if d else None
        valid = c["W"] + c["L"] + c["T"]
        assert c["N"] == valid + c["U"] + c["F"]
        result[track] = {**c, "decisive_win_rate": ratio(c["W"], c["W"] + c["L"]),
                         "tie_adjusted_score": ratio(c["W"] + 0.5 * c["T"], valid),
                         "raw_win_rate": ratio(c["W"], valid), "valid_coverage": ratio(valid, c["N"]),
                         "unknown_rate": ratio(c["U"], c["N"]), "failure_rate": ratio(c["F"], c["N"])}
        track_ids = {p["id"] for p in pairs if p["track"] == track}
        calls = [r for r in attempts if track_ids.intersection(r["ids"])]
        pair_attempts = Counter(i for r in calls for i in r["ids"] if i in track_ids)
        result[track]["request_attempts"] = len(calls)
        result[track]["pair_retries_including_failures"] = sum(max(0, n - 1) for n in pair_attempts.values())
    return result
