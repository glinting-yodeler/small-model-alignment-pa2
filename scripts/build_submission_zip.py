"""Create a lightweight, auditable PA2 handoff ZIP from verified local files."""
import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile


INCLUDE_ROOTS = ("alignment", "configs", "data/persona_lora", "docs", "eval_sets", "kaggle_smoke",
                 "notebooks", "outputs/analysis", "report", "scripts", "submission", "tests")
INCLUDE_FILES = ("README.md", "START_HERE.md", "submission_adapter.py",
                 "pyproject.toml", "requirements.txt", "requirements-kaggle.txt",
                 "requirements-report.txt")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv-local", ".venv-kaggle",
                 "report_render_final"}
PROCESS_DOCS = {"overnight_status.md", "review_improvements.md", "report_outline.md",
                "kaggle_preflight.md", "codex_state.md"}
SCRATCH_SUFFIXES = (".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".synctex.gz", ".tmp")


def build(root, target, report_path):
    from pypdf import PdfReader
    root, target = Path(root).resolve(), Path(target).resolve()
    if not target.is_relative_to(root / "submission"):
        raise ValueError("Submission ZIP must stay inside submission/")
    report = (root / report_path).resolve()
    evidence = root / "submission" / "evidence" / "evidence_manifest.json"
    plots = root / "submission" / "plots_revised"
    if not report.is_file() or not evidence.is_file() or not plots.is_dir() or not list(plots.glob("*.png")):
        raise FileNotFoundError("Final report, evidence manifest and training plots are required")
    pages = PdfReader(str(report)).pages
    if len(pages) > 8:
        raise ValueError(f"Report exceeds eight pages: {len(pages)}")
    files = [root / name for name in INCLUDE_FILES]
    files.extend(p for name in INCLUDE_ROOTS for p in (root / name).rglob("*") if p.is_file())
    selected = sorted({p for p in files if p.is_file() and p != target
                       and not EXCLUDE_PARTS.intersection(p.parts)
                       and not any(part.startswith("report_render_") for part in p.parts)
                       and not any("pre-review-20261015" in part for part in p.parts)
                       and "before-items345-20261015" not in p.name
                       and "pre-analysis-v2-20261016" not in p.name
                       and "outputs/analysis/env_kaggle_download" not in p.relative_to(root).as_posix()
                       and not p.relative_to(root).as_posix().startswith("outputs/analysis/executed_env_")
                        and p.relative_to(root).as_posix() != "docs/codex_state.md"
                       and p.name not in PROCESS_DOCS
                       and "_extracted" not in p.name.casefold()
                       and not p.name.casefold().endswith(SCRATCH_SUFFIXES)
                       and p.name not in {"clean_extract_path.txt", "final_git_status.txt",
                                          "kbench_src_path.txt"}
                       and p.suffix.lower() != ".zip"
                       and (p.suffix.lower() != ".log" or
                            p.relative_to(root).as_posix().startswith("kaggle_smoke/"))
                       and p.suffix.lower() not in {".aux", ".out", ".fls", ".fdb_latexmk"}
                       and p != root / "report" / "main.pdf"
                       and not (p.parent == root / "submission"
                                and p.name.casefold().startswith("pa2_report")
                                and p.suffix.casefold() == ".pdf")
                       and "submission/plots/" not in p.relative_to(root).as_posix()
                       and not p.name.endswith((".pyc", ".pt"))
                       and (p.suffix != ".safetensors" or
                            p.relative_to(root).as_posix() ==
                            "submission/evidence/stages/persona/adapter/adapter_model.safetensors" or
                            p.relative_to(root).as_posix().startswith("submission/evidence_v2/persona/") or
                            p.relative_to(root).as_posix().startswith("submission/evidence_v3/training/"))})
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build from a new disposable staging tree instead of updating an existing
    # archive. Fixed timestamps and sorted relative paths make repeat builds
    # byte-for-byte deterministic.
    with tempfile.TemporaryDirectory(prefix=".pa2_stage_", dir=target.parent) as stage_name:
        stage = Path(stage_name)
        for source in selected:
            relative = source.relative_to(root)
            staged = stage / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)
        temporary = stage / "PA2_Submission.zip"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(p for p in stage.rglob("*") if p.is_file() and p != temporary):
                relative = path.relative_to(stage).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(2026, 10, 17, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=9)
        shutil.copyfile(temporary, target)
    print(f"{target}: {len(selected)} files, {target.stat().st_size} bytes, {len(pages)} report pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="submission/PA2_Submission.zip")
    parser.add_argument("--report", default="submission/PA2_report_final.pdf")
    args = parser.parse_args()
    build(args.root, args.output, args.report)
