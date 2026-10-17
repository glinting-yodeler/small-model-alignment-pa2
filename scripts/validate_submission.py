"""Validate the lightweight final submission archive and evidence hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import re

from pypdf import PdfReader


FORBIDDEN_BASENAMES = {"current_report.txt", "assignment_manual.txt"}
FORBIDDEN_BASENAMES.update({"overnight_status.md", "review_improvements.md",
                            "report_outline.md", "kaggle_preflight.md", "codex_state.md"})
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
                 ".toml", ".tex", ".bib", ".csv", ".ipynb"}
EXPECTED_EVIDENCE_COUNTS = (73, 105, 128)
EXPECTED_BETA_EVIDENCE_COUNT = 38


def scan_archive(zipped: zipfile.ZipFile) -> dict:
    names = zipped.namelist()
    bad_names, boms, mojibake, absolute_paths, identity_strings = [], [], [], [], []
    for name in names:
        base = Path(name).name.casefold()
        parts = Path(name).parts
        scratch = ("_extracted" in base or base.endswith((".aux", ".log", ".out", ".fls",
                   ".fdb_latexmk", ".synctex.gz", ".tmp")) or
                   "__pycache__" in parts or ".pytest_cache" in parts)
        if base.startswith("friend") or base in FORBIDDEN_BASENAMES or scratch:
            bad_names.append(name)
        if Path(name).suffix.casefold() not in TEXT_SUFFIXES:
            continue
        raw = zipped.read(name)
        if raw.startswith(b"\xef\xbb\xbf"):
            boms.append(name)
        text = raw.decode("utf-8", errors="replace")
        if chr(0xC3) + chr(0x192) in text or chr(0xFFFD) in text:
            mojibake.append(name)
        # Model responses may legitimately demonstrate generic filesystem paths or
        # numeric web routes. Preserve those generated artifacts verbatim;
        # path-leak hygiene applies to authored source and metadata.
        is_generation = "/generations/" in f"/{name.replace(chr(92), '/')}"
        if not is_generation and re.search(
                r"(?i)[a-z]:[/\\]users[/\\]|/(?:home|users)/[^/\s]+", text):
            absolute_paths.append(name)
        if "Roll" + " Number" in text or "Due" + " Date:" in text:
            identity_strings.append(name)
    report_pdfs = [n for n in names if Path(n).suffix.casefold() == ".pdf"
                   and "report" in Path(n).name.casefold()]
    return {"forbidden_filenames": sorted(bad_names), "bom_files": sorted(boms),
            "mojibake_files": sorted(mojibake), "absolute_path_files": sorted(absolute_paths),
            "identity_string_files": sorted(identity_strings),
            "report_pdfs": sorted(report_pdfs)}


def validate(root: Path, archive: Path, report: Path) -> dict:
    evidence = root / "submission" / "evidence"
    manifest = json.loads((evidence / "evidence_manifest.json").read_text(encoding="utf-8"))
    bad_hashes = []
    for relative, expected in manifest["files_sha256"].items():
        path = evidence / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            bad_hashes.append(relative)
    evidence_v2 = root / "submission" / "evidence_v2"
    manifest_v2 = json.loads((evidence_v2 / "evidence_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest_v2["files"].items():
        path = evidence_v2 / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            bad_hashes.append(f"evidence_v2/{relative}")
    evidence_v3 = root / "submission" / "evidence_v3"
    manifest_v3 = json.loads((evidence_v3 / "evidence_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest_v3["files"].items():
        path = evidence_v3 / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            bad_hashes.append(f"evidence_v3/{relative}")
    evidence_beta = root / "submission" / "evidence_beta"
    manifest_beta = json.loads((evidence_beta / "evidence_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest_beta["files"].items():
        path = evidence_beta / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            bad_hashes.append(f"evidence_beta/{relative}")

    pages = len(PdfReader(str(report)).pages)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        hygiene = scan_archive(zipped)
    manifest_archive_paths = (
        [f"submission/evidence/{name}" for name in manifest["files_sha256"]]
        + [f"submission/evidence_v2/{name}" for name in manifest_v2["files"]]
        + [f"submission/evidence_v3/{name}" for name in manifest_v3["files"]]
        + [f"submission/evidence_beta/{name}" for name in manifest_beta["files"]]
    )
    missing_manifest_files = sorted(set(manifest_archive_paths).difference(names))
    duplicates = len(names) - len(set(names))
    safetensors = [name for name in names if name.endswith(".safetensors")]
    forbidden = [name for name in names if name.endswith(".zip")
                 or "pre-review-20261015" in name
                 or name == "docs/codex_state.md"
                 or name == "submission/PA2_report.pdf"
                 or name.startswith("submission/plots/")]
    required = {"report/main.tex",
                "submission/evidence/evidence_manifest.json",
                "submission/evidence_v2/evidence_manifest.json",
                "submission/evidence_v3/evidence_manifest.json",
                "submission/evidence_beta/evidence_manifest.json",
                "START_HERE.md", "submission_adapter.py"}
    missing = sorted(required.difference(names))
    expected_weights = sorted([
        "submission/evidence/stages/persona/adapter/adapter_model.safetensors",
        "submission/evidence_v2/persona/v1_reproduction/adapter/adapter_model.safetensors",
        "submission/evidence_v2/persona/v2_lr1e4/adapter/adapter_model.safetensors",
        "submission/evidence_v2/persona/v2_lr2e4/adapter/adapter_model.safetensors",
        "submission/evidence_v3/training/v3a/adapter/adapter_model.safetensors",
        "submission/evidence_v3/training/v3b/adapter/adapter_model.safetensors",
        "submission/evidence_v3/training/v3c/adapter/adapter_model.safetensors",
        "submission/evidence_v3/training/v3d/adapter/adapter_model.safetensors",
    ])
    result = {"evidence_files": len(manifest["files_sha256"]),
              "evidence_v2_files": len(manifest_v2["files"]),
              "evidence_v3_files": len(manifest_v3["files"]),
              "evidence_beta_files": len(manifest_beta["files"]),
              "bad_hashes": bad_hashes, "report_pages": pages,
              "zip_files": len(names), "zip_bytes": archive.stat().st_size,
              "duplicate_entries": duplicates, "safetensors": safetensors,
              "forbidden_entries": forbidden, "missing_required": missing, **hygiene}
    result["missing_manifest_files"] = missing_manifest_files
    hygiene_failed = any(hygiene[key] for key in ("forbidden_filenames", "bom_files",
                         "mojibake_files", "absolute_path_files", "identity_string_files"))
    # The manual lists the report PDF as a separate submission item.  Report
    # source and figures stay in the project archive; generated report PDFs do
    # not, preventing two potentially divergent submission copies.
    if hygiene["report_pdfs"]:
        hygiene_failed = True
    evidence_counts = (result["evidence_files"], result["evidence_v2_files"],
                       result["evidence_v3_files"])
    if (bad_hashes or pages > 8 or duplicates or forbidden or missing
            or missing_manifest_files or evidence_counts != EXPECTED_EVIDENCE_COUNTS
            or result["evidence_beta_files"] != EXPECTED_BETA_EVIDENCE_COUNT
            or safetensors != expected_weights or hygiene_failed):
        raise ValueError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--archive", default="submission/PA2_Submission.zip")
    parser.add_argument("--report", default="submission/PA2_report_final.pdf")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(json.dumps(validate(root, (root / args.archive).resolve(),
                              (root / args.report).resolve()), indent=2))
