"""Build a resumable private Kaggle utility-benchmark judge task."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


COUNTS = {"hh": 300, "ultrafeedback": 300, "persona_style": 20, "persona_quality": 20,
          "hh_beta005_vs_beta01": 300, "hh_beta03_vs_beta01": 300}
RUBRICS = {"hh": "hh", "ultrafeedback": "general", "persona_style": "persona_style",
           "persona_quality": "persona_quality", "hh_beta005_vs_beta01": "hh",
           "hh_beta03_vs_beta01": "hh"}
JUDGE = "google/gemini-3.6-flash"


def main(track):
    if track not in COUNTS:
        raise ValueError(f"Choose one of {list(COUNTS)}")
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_judge_tasks"
    out.mkdir(exist_ok=True)
    target = out / (track + ".py")
    if target.exists() and "Private PA2 evaluation" not in target.read_text(encoding="utf-8")[:100]:
        raise FileExistsError(f"Refusing to replace unexpected judge task: {target}")
    files = [root / name for name in ("alignment/__init__.py", "alignment/io.py",
                                       "alignment/judging.py", "alignment/pairwise.py")]
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(root))
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    template = '''# %%
"""Private PA2 evaluation. Only saved anonymous pair files enter judge prompts."""
import base64, hashlib, io, json, shutil, sys, zipfile
from pathlib import Path
import kaggle_benchmarks as kbench

TRACK = __TRACK_VALUE__
EXPECTED = __EXPECTED_VALUE__
RUBRIC = __RUBRIC_VALUE__
JUDGE_MODEL = __JUDGE_VALUE__
root = Path('/kaggle/working/pa2-judge-source')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(__PAYLOAD_VALUE__))).extractall(root)
sys.path.insert(0, str(root))
from alignment.io import file_hash, read_json, read_jsonl, write_json
from alignment.judging import judge
from alignment.pairwise import summarize

def locate_pairs():
    matches = sorted(Path('/kaggle/input').rglob(TRACK + '_pairs.jsonl'))
    if len(matches) != 1:
        raise ValueError(f'Expected one saved pair file, got {matches}')
    path = matches[0]
    hashes = read_json(path.parent / 'pair_file_hashes.json')
    if file_hash(path) != hashes[path.name]:
        raise ValueError('Pair input hash mismatch')
    pairs = read_jsonl(path)
    if len(pairs) != EXPECTED:
        raise ValueError(f'Expected {EXPECTED} saved pairs, got {len(pairs)}')
    return pairs

# %%
@kbench.task(name=__TASK_NAME__)
def evaluate_saved_pairs(llm):
    pairs = locate_pairs()
    out = Path('/kaggle/working/judging')
    out.mkdir(parents=True, exist_ok=True)
    journal = out / (TRACK + '_judged.jsonl')
    # A resumed task may attach a private dataset containing an earlier journal.
    prior = sorted(Path('/kaggle/input').rglob(journal.name))
    if len(prior) > 1:
        raise ValueError(f'Ambiguous resume journals: {prior}')
    if prior:
        for suffix in ('.jsonl', '.manifest.json', '.attempts.jsonl', '.raw.jsonl', '.failures.jsonl'):
            old = prior[0].parent / (TRACK + '_judged' + suffix)
            if old.is_file():
                shutil.copy2(old, out / old.name)
    actual_judge = kbench.llms[JUDGE_MODEL]
    def backend(prompt):
        with kbench.chats.new('pa2_' + TRACK + '_batch'):
            return actual_judge.prompt(prompt)
    rows = judge(pairs, journal, backend, JUDGE_MODEL, RUBRIC,
                 batch_size=5, sleep_seconds=5, max_attempts=3)
    summary = summarize(pairs, rows, read_jsonl(journal.with_suffix('.attempts.jsonl')))
    write_json(out / (TRACK + '_summary.json'), summary)
    print('JUDGING_COMPLETE', TRACK, summary, flush=True)
    # Kaggle Benchmark tasks require a None return; durable metrics are in the
    # saved JSON artifact and run log.

evaluate_saved_pairs.run(kbench.llm)
'''
    replacements = {"__TRACK_VALUE__": repr(track), "__EXPECTED_VALUE__": str(COUNTS[track]),
                    "__RUBRIC_VALUE__": repr(RUBRICS[track]), "__JUDGE_VALUE__": repr(JUDGE),
                    "__PAYLOAD_VALUE__": repr(payload),
                    "__TASK_NAME__": repr("pa2-judge-" + track.replace("_", "-"))}
    for key, value in replacements.items():
        template = template.replace(key, value)
    target.write_text(template, encoding="utf-8")
    (out / (track + ".source_manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared private benchmark task {target}; no job submitted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/prepare_kaggle_judge_task.py TRACK")
    main(sys.argv[1])
