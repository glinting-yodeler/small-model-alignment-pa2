from pathlib import Path

import pytest


METRICS = (
    Path("submission/evidence/stages/sft/metrics.jsonl"),
    Path("submission/evidence/stages/dpo_hh/metrics.jsonl"),
    Path("submission/evidence/stages/dpo_ultrafeedback/metrics.jsonl"),
    Path("outputs/analysis/persona_comparison.json"),
)


@pytest.fixture(scope="module")
def plotted(tmp_path_factory):
    if not all(path.is_file() for path in METRICS):
        pytest.skip("report metric JSON is unavailable")
    from scripts.report_revision import TEXT_WIDTH_IN, build_report_figures

    root = tmp_path_factory.mktemp("report_figures")
    (root / "figures").mkdir()
    _, diagnostics = build_report_figures(root, close=False)
    yield root, diagnostics, TEXT_WIDTH_IN
    import matplotlib.pyplot as plt
    for item in diagnostics.values():
        plt.close(item["figure"])


def test_figure_pdf_dimensions_match_final_layout(plotted):
    from pypdf import PdfReader

    root, _, text_width = plotted
    for name in ("sft", "hh", "uf", "persona_variants"):
        page = PdfReader(str(root / "figures" / f"{name}.pdf")).pages[0]
        width = float(page.mediabox.width) / 72
        height = float(page.mediabox.height) / 72
        assert width == pytest.approx(text_width, abs=0.05)
        assert height >= 2.0


def test_every_axis_limit_contains_all_plotted_data(plotted):
    _, diagnostics, _ = plotted
    for item in diagnostics.values():
        for axis, values in zip(item["axes"], item["data"]):
            lower, upper = axis.get_ylim()
            assert lower <= min(values)
            assert upper >= max(values)


def test_accuracy_and_persona_margin_extrema_are_visible(plotted):
    _, diagnostics, _ = plotted
    assert diagnostics["hh"]["axes"][2].get_ylim()[1] >= 0.592
    assert diagnostics["uf"]["axes"][2].get_ylim()[1] >= 0.701
    assert diagnostics["persona"]["axes"][1].get_ylim()[0] <= 0
    assert diagnostics["persona"]["axes"][1].get_ylim()[1] >= 8.918
    assert diagnostics["persona"]["axes"][0].get_ylim()[1] >= 3.301
