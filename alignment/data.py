"""Strict schema conversion, prompt-disjoint subsets, immutable manifests."""
from collections import Counter
import random
import re
import unicodedata
from .io import digest, write_json


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty_or_nonstring")
    return value.strip()


def context(messages):
    roles = [m.get("role") for m in messages]
    if roles == ["user"]:
        return nonempty(messages[0]["content"])
    if roles == ["system", "user"]:
        return "System: " + nonempty(messages[0]["content"]) + "\n\n" + nonempty(messages[1]["content"])
    raise ValueError("multiturn_or_invalid_context")


def parse_sft(row):
    messages = row["messages"]
    offset = int(bool(messages) and messages[0].get("role") == "system")
    if len(messages) < offset + 2 or messages[offset + 1].get("role") != "assistant":
        raise ValueError("invalid_first_exchange")
    return {"prompt": context(messages[:offset + 1]),
            "response": nonempty(messages[offset + 1]["content"])}


def hh_branch(text):
    parts = re.split(r"\n\n(Human|Assistant):", text)
    if len(parts) != 5 or parts[0].strip() or parts[1] != "Human" or parts[3] != "Assistant":
        raise ValueError("multiturn_or_invalid_hh")
    return nonempty(parts[2]), nonempty(parts[4])


def parse_preference(row, kind):
    if kind == "hh":
        prompt, chosen = hh_branch(row["chosen"])
        other, rejected = hh_branch(row["rejected"])
        if prompt != other:
            raise ValueError("mismatched_context")
    elif isinstance(row["chosen"], list):
        c, r = row["chosen"], row["rejected"]
        if not c or not r or c[-1].get("role") != "assistant" or r[-1].get("role") != "assistant":
            raise ValueError("invalid_assistant")
        if c[:-1] != r[:-1]:
            raise ValueError("mismatched_context")
        prompt = context(c[:-1])
        if row.get("prompt") and nonempty(row["prompt"]) != prompt:
            raise ValueError("mismatched_explicit_prompt")
        chosen, rejected = nonempty(c[-1]["content"]), nonempty(r[-1]["content"])
    else:
        prompt, chosen, rejected = (nonempty(row[k]) for k in ("prompt", "chosen", "rejected"))
    if normalize(chosen) == normalize(rejected):
        raise ValueError("identical_responses")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def select_splits(records, kind, train_size, val_size, seed=42, excluded=(), source="local",
                  validator=None):
    if train_size < 1 or val_size < 1:
        raise ValueError("positive train and validation sizes required")
    blocked = {normalize(p) for p in excluded}
    seen, valid, counts = set(), [], Counter()
    # One retained record per normalized prompt makes exact prompt-disjoint counts possible.
    for index, row in enumerate(records):
        counts["raw"] += 1
        try:
            parsed = parse_sft(row) if kind == "sft" else parse_preference(row, kind)
            key = normalize(parsed["prompt"])
            if key in blocked:
                raise ValueError("evaluation_overlap")
            if key in seen:
                raise ValueError("duplicate_prompt")
            if validator is not None and not validator(parsed):
                raise ValueError("unscoreable_after_truncation")
            seen.add(key)
            parsed.update(id=f"{kind}-{digest([source, index, parsed])[:20]}", source=source,
                          source_index=index)
            valid.append(parsed)
            counts["valid"] += 1
        except (ValueError, KeyError, TypeError, IndexError) as e:
            counts[str(e) if isinstance(e, ValueError) else "malformed_schema"] += 1
    if len(valid) < train_size + val_size:
        raise ValueError(f"Need {train_size + val_size} usable unique prompts; counts={dict(counts)}")
    random.Random(seed).shuffle(valid)
    train, val = valid[:train_size], valid[train_size:train_size + val_size]
    return train, val, dict(counts)


def save_splits(output, train, val, metadata):
    from pathlib import Path
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace prepared data: {out}")
    write_json(out / "train.json", train)
    write_json(out / "validation.json", val)
    write_json(out / "manifest.json", {**metadata, "train_count": len(train), "validation_count": len(val),
                                       "train_hash": digest(train), "validation_hash": digest(val)})
