from scripts.build_report_tex import tex


def test_report_tex_escapes_table_metacharacters():
    assert tex("a_b&10%") == r"a\_b\&10\%"


def test_generated_report_numbers_match_saved_evidence():
    from scripts.check_report_numbers import check
    assert check()["status"] == "ok"


def test_compiled_report_has_no_macro_spacing_artifacts():
    import re
    from pathlib import Path
    import pytest
    pdf = Path("report/main.pdf")
    if not pdf.exists():
        pytest.skip("compiled report is optional in source-only test environments")
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
    patterns = (r"v3[a-d][a-z]", r"\d+/\d+[a-z]", r"\bfor(?:style|quality)\b", r"\?\?")
    assert not any(re.search(pattern, text) for pattern in patterns)
