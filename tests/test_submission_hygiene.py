import io
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.validate_submission import scan_archive


def scan(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        return scan_archive(archive)


def test_clean_archive_hygiene():
    result = scan({"README.md": "clean relative documentation\n"})
    assert result["report_pdfs"] == []
    assert not any(result[key] for key in ("forbidden_filenames", "bom_files",
                   "mojibake_files", "absolute_path_files", "identity_string_files"))


@pytest.mark.parametrize(("name", "content", "field"), [
    ("outputs/friend_report.txt", "x", "forbidden_filenames"),
    ("notes.md", b"\xef\xbb\xbftext", "bom_files"),
    ("notes.md", "T4" + chr(0xC3) + chr(0x192) + "bad", "mojibake_files"),
    ("notes.md", "C:" + "\\Users\\student\\file", "absolute_path_files"),
    ("notes.md", "/" + "home/student/file", "absolute_path_files"),
    ("notes.md", "Roll" + " Number 123", "identity_string_files"),
    ("notes.md", "Due" + " Date: tomorrow", "identity_string_files"),
    ("report/main_extracted.txt", "scratch", "forbidden_filenames"),
    ("report/main.aux", "scratch", "forbidden_filenames"),
    ("pkg/__pycache__/x.pyc", b"x", "forbidden_filenames"),
    ("docs/overnight_status.md", "process", "forbidden_filenames"),
])
def test_archive_hygiene_detects_forbidden_content(name, content, field):
    result = scan({name: content})
    assert name in result[field]


def test_all_packaged_evidence_manifests_are_complete_and_hash_exact():
    roots = (
        (Path("submission/evidence"), "files_sha256", 73),
        (Path("submission/evidence_v2"), "files", 105),
        (Path("submission/evidence_v3"), "files", 128),
        (Path("submission/evidence_beta"), "files", 38),
    )
    for root, key, expected_count in roots:
        manifest = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))[key]
        assert len(manifest) == expected_count
        assert not any(Path(relative).suffix.lower() == ".log" for relative in manifest)
        for relative, expected_hash in manifest.items():
            path = root / relative
            assert path.is_file(), relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, relative


def test_final_archive_has_unique_entries_and_every_manifest_member():
    archive = Path("submission/PA2_Submission.zip")
    if not archive.exists():
        pytest.skip("final archive is built during the packaging gate")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert len(names) == len(set(names))
        members = set(names)
        for root, key in (("evidence", "files_sha256"),
                          ("evidence_v2", "files"),
                          ("evidence_v3", "files"),
                          ("evidence_beta", "files")):
            manifest_name = f"submission/{root}/evidence_manifest.json"
            manifest = json.loads(zipped.read(manifest_name))[key]
            assert not any(Path(relative).suffix.lower() == ".log" for relative in manifest)
            assert all(f"submission/{root}/{relative}" in members for relative in manifest)

