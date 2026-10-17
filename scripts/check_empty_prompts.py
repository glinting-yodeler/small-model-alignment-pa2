"""Audit prepared splits for empty source and formatted prompts."""
import json
from pathlib import Path

from alignment.formatting import format_prompt


def main():
    rows = []
    for track in ("sft", "hh", "ultrafeedback"):
        for split in ("train", "validation"):
            path = Path("data/prepared") / track / f"{split}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            source_empty = sum(not str(item.get("prompt", "")).strip() for item in data)
            formatted_empty = sum(not format_prompt(str(item.get("prompt", ""))) for item in data)
            rows.append({"track": track, "split": split, "path": path.as_posix(),
                         "examples": len(data), "empty_source_prompts": source_empty,
                         "empty_formatted_strings": formatted_empty})
    result = {"method": "exact scan of frozen prepared JSON; tokenizer-empty prompts are additionally counted at training encode time",
              "splits": rows, "empty_source_total": sum(r["empty_source_prompts"] for r in rows),
              "empty_formatted_string_total": sum(r["empty_formatted_strings"] for r in rows)}
    out = Path("outputs/analysis/empty_prompt_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
