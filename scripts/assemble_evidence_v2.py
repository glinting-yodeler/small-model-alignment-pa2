"""Assemble additive v2 evidence without changing the preserved v1 bundle."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "submission" / "evidence_v2"


def copy_file(src, dst):
    target = DEST / dst
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / src, target)


def copy_tree(src, dst, *, exclude=()):
    source, target = ROOT / src, DEST / dst
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*exclude))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    # Three compact adapters and their direct training evidence. Checkpoint copies
    # are excluded because each final adapter is byte-equivalent evidence.
    for source_name, dest_name in (("persona_v1_reproduction", "v1_reproduction"),
                                   ("persona_v2", "v2_lr2e4"),
                                   ("persona_v2_lr1e4", "v2_lr1e4")):
        copy_tree(f"outputs/persona_v2/outputs/{source_name}", f"persona/{dest_name}",
                  exclude=("checkpoint-*",))
    copy_tree("outputs/persona_v2_generation", "persona/generation")
    copy_tree("outputs/persona_v2_evaluation", "persona/evaluation")
    copy_tree("outputs/analysis/env_kaggle", "environment")
    for path in ("analysis.json", "persona_comparison.json", "empty_prompt_audit.json"):
        copy_file(f"outputs/analysis/{path}", f"analysis/{path}")
    beta = ROOT / "outputs/analysis/hh_beta_pilots.json"
    if beta.exists():
        copy_tree("outputs/hh_beta_pilots_v2", "hh_beta_pilots")
        copy_file("outputs/analysis/hh_beta_pilots.json", "analysis/hh_beta_pilots.json")
    for path in ("persona_lora_v2.yaml", "persona_lora_v2_lr1e4.yaml",
                 "generate_persona_lora_v2.yaml", "generate_persona_lora_v2_lr1e4.yaml",
                 "dpo_hh_pilot_beta005.yaml", "dpo_hh_pilot_beta01.yaml", "dpo_hh_pilot_beta03.yaml"):
        copy_file(f"configs/{path}", f"configs/{path}")
    # The submission archive intentionally omits runtime logs. Keep the original
    # local logs as provenance, but do not advertise files that cannot be present
    # in the packaged evidence bundle.
    manifest = {p.relative_to(DEST).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(DEST.rglob("*"))
                if p.is_file() and p.name != "evidence_manifest.json"
                and p.suffix.lower() != ".log"}
    (DEST / "evidence_manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest), "bytes": sum((DEST / p).stat().st_size for p in manifest)}))


if __name__ == "__main__":
    main()
