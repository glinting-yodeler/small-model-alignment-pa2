"""Build one resumable Kaggle Benchmark task for all persona-v3 comparisons."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

TRACKS = tuple(f"persona_v3{letter}_{rubric}" for letter in "abcd"
               for rubric in ("style", "quality"))
JUDGE = "google/gemini-3.6-flash"


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_judge_tasks" / "persona_v3"
    out.mkdir(parents=True, exist_ok=True)
    files = [root / name for name in ("alignment/__init__.py", "alignment/io.py",
                                       "alignment/judging.py", "alignment/pairwise.py")]
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(root))
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    template = '''# %%
"""Private PA2 persona-v3 evaluation; anonymous saved pairs only."""
import base64, io, shutil, sys, time, zipfile
from pathlib import Path
import kaggle_benchmarks as kbench
root=Path('/kaggle/working/pa2-persona-v3-judge-source'); root.mkdir(parents=True,exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(__PAYLOAD__))).extractall(root); sys.path.insert(0,str(root))
from alignment.io import file_hash, read_json, read_jsonl, write_json
from alignment.judging import judge
from alignment.pairwise import summarize
TRACKS=__TRACKS__; JUDGE_MODEL=__JUDGE__

def locate(track):
    matches=sorted(Path('/kaggle/input').rglob(track+'_pairs.jsonl'))
    if len(matches)!=1: raise ValueError(f'{track}: expected one pair file, got {matches}')
    path=matches[0]; hashes=read_json(path.parent/'pair_file_hashes.json')
    if file_hash(path)!=hashes[path.name]: raise ValueError(f'{track}: hash mismatch')
    pairs=read_jsonl(path)
    if len(pairs)!=20 or len({p['id'] for p in pairs})!=20: raise ValueError(f'{track}: invalid coverage')
    return pairs

# %%
@kbench.task(name='pa2-judge-persona-v3')
def evaluate_saved_pairs(llm):
    out=Path('/kaggle/working/judging'); out.mkdir(parents=True,exist_ok=True)
    actual_judge=kbench.llms[JUDGE_MODEL]
    for index,track in enumerate(TRACKS):
        pairs=locate(track); journal=out/(track+'_judged.jsonl')
        prior=sorted(Path('/kaggle/input').rglob(journal.name))
        if len(prior)>1: raise ValueError(f'{track}: ambiguous resume journals')
        if prior:
            for suffix in ('.jsonl','.manifest.json','.attempts.jsonl','.raw.jsonl','.failures.jsonl'):
                old=prior[0].parent/(track+'_judged'+suffix)
                if old.is_file(): shutil.copy2(old,out/old.name)
        if index: time.sleep(5)
        def backend(prompt, track=track):
            with kbench.chats.new('pa2_'+track+'_batch'):
                return actual_judge.prompt(prompt)
        rubric='persona_style' if track.endswith('_style') else 'persona_quality'
        rows=judge(pairs,journal,backend,JUDGE_MODEL,rubric,
                   batch_size=5,sleep_seconds=5,max_attempts=3)
        summary=summarize(pairs,rows,read_jsonl(journal.with_suffix('.attempts.jsonl')))
        write_json(out/(track+'_summary.json'),summary)
        print('JUDGING_COMPLETE',track,summary,flush=True)

evaluate_saved_pairs.run(kbench.llm)
'''
    template = template.replace("__PAYLOAD__", repr(payload)).replace(
        "__TRACKS__", repr(TRACKS)).replace("__JUDGE__", repr(JUDGE))
    (out / "persona_v3.py").write_text(template, encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    print(out / "persona_v3.py")


if __name__ == "__main__":
    main()
