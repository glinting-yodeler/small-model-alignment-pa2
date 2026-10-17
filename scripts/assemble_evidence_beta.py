"""Assemble hash-manifested generation-level HH beta evidence."""
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission/evidence_beta"


def copy(src, dst):
    target = DEST / dst; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, target)


def main():
    if DEST.exists(): shutil.rmtree(DEST)
    for label in ("beta005", "beta01", "beta03"):
        for suffix in (".json", ".jsonl", ".manifest.json"):
            copy(f"outputs/hh_beta_generation/outputs/generations/{label}{suffix}", f"generations/{label}{suffix}")
        copy(f"outputs/hh_beta_generation/outputs/sanity/{label}.json", f"sanity/{label}.json")
        for name in ("completion.json", "manifest.json", "metrics.jsonl", "resolved_config.json"):
            copy(f"outputs/hh_beta_generation/outputs/training/{label}/{name}", f"training/{label}/{name}")
    copy("outputs/hh_beta_generation/outputs/beta_generation_summary.json", "beta_generation_summary.json")
    copy("outputs/analysis/hh_beta_generation.json", "analysis.json")
    (DEST / "judging").mkdir(parents=True, exist_ok=True)
    for track in ("hh_beta005_vs_beta01", "hh_beta03_vs_beta01"):
        copy(f"outputs/hh_beta_pairs/{track}_pairs.jsonl", f"pairs/{track}_pairs.jsonl")
        source = next(ROOT.glob(f"outputs/hh_beta_judging/**/judging/{track}_judged.jsonl")).parent
        for suffix in (".jsonl", ".manifest.json", ".attempts.jsonl", ".raw.jsonl", ".failures.jsonl"):
            path = source / f"{track}_judged{suffix}"
            if path.is_file(): shutil.copy2(path, DEST / "judging" / path.name)
        shutil.copy2(source / f"{track}_summary.json", DEST / "judging" / f"{track}_summary.json")
    files = {p.relative_to(DEST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(DEST.rglob("*")) if p.is_file()}
    (DEST / "evidence_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": sum((DEST / p).stat().st_size for p in files)}))


if __name__ == "__main__":
    main()
