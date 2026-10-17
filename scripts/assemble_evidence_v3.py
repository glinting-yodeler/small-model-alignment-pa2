"""Assemble additive persona-v3 evidence without modifying earlier evidence."""
import hashlib
import json
from pathlib import Path
import shutil


def copy(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    root = Path(__file__).resolve().parents[1]
    dest = root / "submission" / "evidence_v3"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to replace existing evidence: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    run = root / "outputs" / "persona_v3" / "outputs"
    for name in ("v3a", "v3b", "v3c", "v3d"):
        for relative in ("manifest.json", "metrics.json", "reload_verified.json",
                         "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            copy(run / name / relative, dest / "training" / name / relative)
        copy(root / "configs" / f"persona_lora_{name}.yaml",
             dest / "configs" / f"persona_lora_{name}.yaml")
        copy(root / "configs" / f"generate_persona_{name}.yaml",
             dest / "configs" / f"generate_persona_{name}.yaml")
        for stem in (f"heldout_{name}", f"persona_{name}"):
            for suffix in (".json", ".jsonl", ".manifest.json"):
                copy(run / "generations" / f"{stem}{suffix}",
                     dest / "generations" / f"{stem}{suffix}")
        copy(run / "sanity" / f"sanity_{name}.json", dest / "sanity" / f"sanity_{name}.json")
    for name in ("base", "v1", "v2_lr2e4", "v2_lr1e4"):
        for suffix in (".json", ".jsonl", ".manifest.json"):
            copy(run / "generations" / f"heldout_{name}{suffix}",
                 dest / "generations" / f"heldout_{name}{suffix}")
    for name in ("environment.json", "persona_heldout_22.json", "persona_v3_summary.json"):
        copy(run / name, dest / "run" / name)
    for name in ("persona_selection_v2.json", "persona_selection_v2_catalog.json",
                 "persona_v3_judging_summary.json", "persona_comparison.json", "report_examples.json"):
        copy(root / "outputs" / "analysis" / name, dest / "analysis" / name)
    for path in sorted((root / "outputs" / "persona_v3_pairs").glob("*.jsonl")):
        copy(path, dest / "pairs" / path.name)
    judge_roots = list((root / "outputs" / "persona_v3_judging").rglob("judging"))
    if len(judge_roots) != 1:
        raise ValueError(f"Expected one judging directory, got {judge_roots}")
    for path in sorted(judge_roots[0].glob("*")):
        if path.is_file():
            copy(path, dest / "judging" / path.name)
    run_root = judge_roots[0].parent
    for path in run_root.glob("*.json"):
        copy(path, dest / "judging_run" / path.name)
    copy(root / "kaggle_judge_tasks" / "persona_v3" / "persona_v3.py",
         dest / "judging_run" / "persona_v3.py")
    copy(root / "docs" / "persona_selection_rule_v2.md", dest / "persona_selection_rule_v2.md")
    files = {}
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            files[path.relative_to(dest).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest / "evidence_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({"files": len(files), "bytes": sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}))


if __name__ == "__main__":
    main()
