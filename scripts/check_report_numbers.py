"""Check that generated LaTeX values agree with saved machine-readable evidence."""
import json
import re
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check():
    analysis = read("outputs/analysis/analysis.json")
    persona = read("outputs/analysis/persona_comparison.json")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             Path("report/generated_numbers.tex").read_text(encoding="utf-8")))
    macros = {key: value.removesuffix(r"\xspace") for key, value in macros.items()}
    expected = {
        "HHW": str(analysis["evaluations"]["hh"]["win"]),
        "HHL": str(analysis["evaluations"]["hh"]["loss"]),
        "HHT": str(analysis["evaluations"]["hh"]["tie"]),
        "HHScore": f'{100*analysis["evaluations"]["hh"]["tie_adjusted"]:.1f}',
        "HHUniqueN": str(analysis["hh"]["unique_prompts"]["n"]),
        "HHUniqueScore": f'{100*analysis["hh"]["unique_prompts"]["tie_adjusted"]:.1f}',
        "UFScore": f'{100*analysis["evaluations"]["ultrafeedback"]["tie_adjusted"]:.1f}',
    }
    assert all(macros.get(k) == v for k, v in expected.items()), (macros, expected)
    tables = Path("report/generated_tables.tex").read_text(encoding="utf-8")
    headline = Path("report/generated_headline.tex").read_text(encoding="utf-8")
    for row in persona["runs"]:
        assert f"{row['held_out_reward_accuracy']:.3f}" in tables
        assert f"{row['held_out_reward_margin']:.3f}" in tables
        assert f"{row['eval_loss']:.3f}" in tables
        assert f"{row['outputs_changed_vs_base']}/{row['n']}" in tables
        assert f"{row['degeneration_pct']:.1f}/{row['identical_to_base_pct']:.1f}" in tables
        if row['style']:
            assert f"{row['style']['W']}/{row['style']['L']}/{row['style']['T']}" in tables
            assert f"{100*row['style']['tie_adjusted']:.1f}" in tables
        if row['quality']:
            assert f"{row['quality']['W']}/{row['quality']['L']}/{row['quality']['T']}" in tables
            assert f"{100*row['quality']['tie_adjusted']:.1f}" in tables
    selected = next(row for row in persona['runs'] if row['run'] == persona['revised_selected'])
    for rubric in ('style', 'quality'):
        row = selected[rubric]
        assert f"{row['W']}/{row['L']}/{row['T']}" in headline
        assert f"{100*row['tie_adjusted']:.1f}\\%" in headline
    beta = Path("outputs/analysis/hh_beta_pilots.json")
    if beta.exists():
        for run in read(beta)["runs"]:
            for point in run["validation"]:
                assert f"{point['reward_margin']:.3f}" in tables
    beta_generation = Path("outputs/analysis/hh_beta_generation.json")
    if beta_generation.exists():
        diagnostics = Path("report/generated_diagnostics.tex").read_text(encoding="utf-8")
        for row in read(beta_generation)["comparisons"].values():
            assert f"{row['win']}/{row['loss']}/{row['tie']}" in diagnostics
            assert f"{100*row['tie_adjusted']:.1f}" in diagnostics
    examples = read("outputs/analysis/report_examples.json")
    notes = read("report/example_notes.json")
    displayed_ids = {saved["id"] for rows in examples.values() for saved in rows.values()}
    assert set(notes) == displayed_ids
    assert all(set(value) == {"note"} and value["note"].strip() for value in notes.values())
    try:
        from scripts.select_report_examples import excerpt, load_jsonl
    except ModuleNotFoundError:  # direct execution from the scripts directory
        from select_report_examples import excerpt, load_jsonl
    sources = {
        "hh": Path("submission/evidence/evaluation/hh_pairs.jsonl"),
        "ultrafeedback": Path("submission/evidence/evaluation/ultrafeedback_pairs.jsonl"),
        "persona_v3b": Path("submission/evidence_v3/pairs/persona_v3b_style_pairs.jsonl"),
    }
    for track, rows in examples.items():
        pairs = {row['id']: row for row in load_jsonl(sources[track])}
        for outcome, saved in rows.items():
            pair = pairs[saved['id']]
            candidate_side = next(side for side, model in pair['mapping'].items()
                                  if model == pair['candidate'])
            baseline_side = 'B' if candidate_side == 'A' else 'A'
            limit = 360 if track == "hh" and outcome == "win" else 300
            assert saved['candidate'] == excerpt(pair[candidate_side], limit)
            assert saved['baseline'] == excerpt(pair[baseline_side], limit)
            assert saved['prompt'] == excerpt(pair['instruction'])
    pdf = Path("report/main.pdf")
    if pdf.exists():
        from pypdf import PdfReader
        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        for rows in examples.values():
            for saved in rows.values():
                assert saved['prompt_id'] in pdf_text
        assert persona['revised_selected'] in pdf_text
        bad_patterns = (r"v3[a-d][a-z]", r"\d+/\d+[a-z]", r"\bfor(?:style|quality)\b", r"\?\?",
                        r"\d+(?:\.\d+)?\s+%", r"\d+(?:\.\d+)?\s+\]\s*%",
                        r"\(inconclusive\s+\(")
        assert not any(re.search(pattern, pdf_text) for pattern in bad_patterns), pdf_text
        normalized_pdf = " ".join(pdf_text.split())
        for note in notes.values():
            assert " ".join(note["note"].split()) in normalized_pdf
    behavior = analysis["length_and_repetition"]
    for key in ("alpaca_base", "alpaca_sft", "alpaca_uf", "hh_sft", "hh_dpo", "persona_base"):
        row = behavior[key]
        assert f"{row['mean_words']:.1f}" in Path("report/generated_results.tex").read_text(encoding="utf-8")
    for row in analysis["hh"]["by_category"].values():
        assert f"{row.get('win',0)}&{row.get('loss',0)}&{row.get('tie',0)}" in Path("report/generated_results.tex").read_text(encoding="utf-8")
    return {"status": "ok", "macros_checked": len(expected),
            "persona_runs_checked": len(persona["runs"]), "beta_checked": beta.exists(),
            "beta_generation_checked": beta_generation.exists(),
            "headline_rows_checked": 5,
            "quoted_examples_checked": sum(len(rows) for rows in examples.values()),
            "example_notes_checked": len(notes),
            "pdf_text_checked": pdf.exists()}


def main():
    print(json.dumps(check()))


if __name__ == "__main__":
    main()
