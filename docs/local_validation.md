# Local validation — 12 October 2026

All tests run with Windows CPython 3.12 in `.venv-local`, CPU PyTorch, and
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. No pretrained weights or datasets
are downloaded by tests. Exact installed versions are in `local_environment.json`.

| Check | Observed result |
|---|---|
| Unchanged public adapter suite | 7 passed |
| Added core, pipeline and real-library tests | 39 passed |
| Final combined suite | 46 passed in 37.52 seconds, no failures |
| `pip check` | No broken requirements |
| Nine workflow script `--help` entry points | All pass |
| `compileall` and `git diff --check` | Pass |
| Pilot archive/embedded bootstrap | Source hashes and Python syntax verified; never executed or submitted |
| Preservation | All 19 original files exist; 15 unmodified files match starting SHA-256; four intended files changed |

The four intentionally changed originals are `.gitignore`, `README.md`,
`pyproject.toml`, and `submission_adapter.py`; their originals remain in the
pre-implementation backup. Public tests, toy model, requirements, supplied
data/evaluation files, PDF, plan and all smoke evidence remain unchanged.

Added coverage includes EOS/truncation boundaries, PAD=EOS and absent special
tokens, positional padding, response masks, row order, explicit causal scoring,
zero masks, scalar reductions, DPO numerical extremes and analytic gradients,
reference eval/no-grad/freeze, real optimizer updates, exact token accumulation,
partial windows, exact CPU resume, schema rejection and prompt-disjoint splits,
blind mappings, interrupted/partial judge resume and throttling, failures and
denominators. Tiny real Transformers models exercise custom SFT and both DPO
branches, checkpoint lineage/reload, generation resume, TRL/PEFT training and
adapter reload. Plotting uses metrics from a tiny real CPU training test.

A generation integration test initially failed because tokenizer `token_type_ids`
were passed to a model that rejects them. Production generation was fixed to
request only causal-LM inputs; the test was retained and then passed. No public
test or supplied dataset was changed to obtain passing results.

The plotting check reports third-party Pyparsing deprecation warnings from
Matplotlib 3.10.1; the PNG is successfully written. No warnings were suppressed.

## Later validation — 13 October 2026

After fixing JSON numeric config parsing, nested Kaggle input location,
same-stage package imports, fp32 persona stability, and strict fenced JSON
judge parsing, the unchanged public suite and all added tests were rerun in
`.venv-local`: **51 passed, 14 third-party Matplotlib deprecation warnings**
in 126.73 seconds. The seven original public tests remain unchanged. The added
fenced-JSON case accepts an exact outer `json` code fence and rejects prose.
The integrated T4×2 pilot also passed its packaged tests and checkpoint audits;
see `docs/experiment_log.md` and its downloaded artifact for remote evidence.
The suites were rerun after the report, DPO artifact, and generation dependency
work: the combined suite passed **51/51** in 37.80 seconds. Separately, the
unchanged public file passed **7/7** in 8.34 seconds and the three added test
files passed **44/44** in 30.47 seconds with the same 14 Matplotlib warnings.

Full data counts, GPU execution, SFT/persona training, and persona judgments
were subsequently verified as recorded in `docs/experiment_log.md`. Both full
DPO jobs, fixed-set generation, and all main-track judgments completed.

Final packaging validation on 14 October 2026 passed **51/51** tests in 47.91
seconds; the unchanged public suite separately passed **7/7** in 9.78 seconds.
`compileall`, `git diff --check`, all 68 evidence-manifest hashes, ZIP duplicate
checks, the five-page report limit, and the large-weight exclusion check passed.
The same 14 third-party Matplotlib/Pyparsing deprecation warnings remain.

## Final analytical review — 15 October 2026

After revising the report and evidence package, the unchanged public test file
passed **7/7** in 248.17 seconds and the three added test files passed **44/44**
in 933.42 seconds. The unusually long CPU times reflect concurrent real-library
integration tests; there were no failures. The added suite emitted the same 14
third-party Matplotlib/Pyparsing deprecation warnings. `compileall` and
`git diff --check` passed. The final package validation checks 73 evidence
hashes, a seven-page PDF, archive uniqueness, and exclusion of full-model
weights, optimizer states, superseded artifacts, and nested ZIP files.

## Kernel identifier and report corrections — 15 October 2026

The focused local package-builder regression and unchanged public adapter tests
passed **9/9** in 9.75 seconds. The core, real-library, pipeline, and focused
builder tests passed **46/46** in 36.73 seconds, with the same 14 third-party
Matplotlib/Pyparsing deprecation warnings. The two DPO training packages were
regenerated locally and their metadata IDs matched the corresponding generation
kernel sources. No package was uploaded or run remotely. The report was rebuilt
from saved evidence, remained seven pages, and every page was visually inspected.

## Final revision validation — 16 October 2026

The unchanged public adapter tests passed **7/7** in 10.53 seconds. The exact
requested full command, `PYTHONPATH=. python -m pytest -q`, produced **58 passed,
1 failed, 14 warnings** in 36.76 seconds. The sole failure is
`test_preserved_public_tests_data_pdf_and_plan_hashes`: its manifest requires the
original `requirements.txt` SHA-256, while revision item C explicitly requires
adding `trl` and `peft` to that file. The test and manifest were not changed to
hide this conflict. The warnings are the previously recorded third-party
Matplotlib/Pyparsing deprecations.

The revised report builds from saved artifacts at exactly eight pages. Python
compilation, YAML/JSON notebook parsing, and `git diff --check` passed. Persona
v2 results are documented from completed runs. The earlier note that the
optional HH beta pilots were prepared-only is superseded by the completed
generation check documented below. No submitted checkpoint was replaced.

## Five-item manual compliance pass — 16 October 2026

The repository suite passed **60/60** with 14 third-party
Matplotlib/Pyparsing deprecation warnings in 36.97 seconds. The original
requirements hash remains immutable in `docs/starter_manifest.json`; the test
asserts the exact approved dependency delta. Quantitative analysis was
recomputed from unchanged saved generations and judgments and retained HH
126/90/84, UltraFeedback 86/132/82, persona style 0/1/19, and persona
quality/safety 1/3/16. The report rebuilt at eight pages and all pages were
rendered and visually inspected without clipping, overlap, broken tables, or
unreadable figures.

The final ZIP was extracted to a new GUID-named temporary directory and its
complete included suite passed **60/60** with the same 14 third-party warnings
in 33.38 seconds. This verifies that the archive contains the plan, smoke files,
supplied datasets/evaluation files, public tests, toy model, and all local code
needed by its tests.

## 2026-10-16 persona-v3 Session 1

Command actually run from the repository root:

```powershell
$env:PYTHONPATH='.'; .\.venv-local\Scripts\python.exe -m pytest -q
```

Result: **74 passed**, with 14 third-party Matplotlib/PyParsing deprecation warnings, in 45.79 seconds on the final Session-1 rerun. The unchanged public adapter tests are included in this count. New coverage includes the revised persona selection rule, report generators/checker, and archive hygiene rejection cases.

## Final persona-v3 packaging validation ? 16 October 2026

After the completed persona-v3 generation and blinded judging runs, the unchanged
public adapter suite passed **7/7** in 10.40 seconds. The complete offline suite
passed **74/74** with 14 third-party Matplotlib/PyParsing deprecation warnings in
41.80 seconds. The report provenance checker passed all seven persona-run rows,
five headline rows, nine exact quoted examples, beta diagnostics, and PDF text
checks. The final report is seven A4 pages and its LaTeX log contains no overfull,
underfull, undefined-reference, package, missing-character, or LaTeX warnings.

The final archive was extracted into a new directory and its complete included
suite passed **74/74** with the same 14 third-party warnings in 46.35 seconds.
This is a historical, superseded validation record: it reported 309 manifest
entries (73 v1, 108 v2, 128 v3), but the v2 count included three runtime logs
that the ZIP policy excluded. The corrected final-package validation appears
below.

## Task 8 report revision — 17 October 2026

The exact command `$env:PYTHONPATH='.'; .\.venv-local\Scripts\python.exe -m
pytest -q` passed **75/75** tests in 33.42 seconds on Python 3.12 with CPU
PyTorch 2.6.0. The 14 warnings are third-party Matplotlib/PyParsing deprecation
warnings. The unchanged seven-test public adapter suite is included. The report
evidence checker passed all headline, persona, response-behavior, HH-category,
example-text, and example-note checks. The final eight-page LaTeX build has no
overfull/underfull box, undefined-reference, package, glyph, or LaTeX warnings.

## Final grading cleanup — 17 October 2026

At this cleanup stage, local copies of `notebooks/00_setup_and_env_check.ipynb`
had no saved outputs and no executed artifact was available, so none was
fabricated. This historical state is superseded by the authentic v5 execution
record below. The package retains the real
machine-readable Kaggle environment evidence and excludes the internal
`docs/codex_state.md` process note. The report restores DPO reward/margin plots,
checkpoint-selection and convergence discussion, and an evidence-based account
of the small nonzero initial pair margins.

The unchanged public suite passed **7/7** in 7.50 seconds. The complete offline
suite passed **75/75** in 44.85 seconds with the same 14 third-party
Matplotlib/PyParsing deprecation warnings.

The earlier local-root validator run reported 490 unique files and 309 evidence
entries. That result is superseded because three v2 runtime logs existed locally
but were excluded from the archive. `docs/codex_state.md` was absent from that
archive. The corrected clean-extraction result appears below.

## Corrected evidence manifest and clean archive — 17 October 2026

The v2 evidence assembler was rerun after excluding `.log` files from manifest
generation. It produced exactly 105 entries; the three original runtime logs
remain locally as provenance and are neither manifest entries nor ZIP members.
The final evidence counts are **73 v1 + 105 v2 + 128 v3 = 306**.

Commands actually run from the repository root:

```powershell
$env:PYTHONPATH='.'
.\.venv-local\Scripts\python.exe scripts\assemble_evidence_v2.py
.\.venv-local\Scripts\python.exe scripts\build_submission_zip.py
.\.venv-local\Scripts\python.exe scripts\validate_submission.py
.\.venv-local\Scripts\python.exe scripts\check_report_numbers.py
.\.venv-local\Scripts\python.exe -m compileall -q alignment scripts submission_adapter.py
.\.venv-local\Scripts\python.exe -m pytest -q tests\test_public_adapter_smoke.py
.\.venv-local\Scripts\python.exe -m pytest -q
```

Observed results: public tests **7/7** in 7.57 seconds; complete suite **77/77**
in 34.58 seconds with 14 third-party Matplotlib/PyParsing deprecation warnings;
compilation and the report-number checker passed. The three report copies were
byte-identical, eight-page PDFs with SHA-256
`197ab164c691c80a13890ae581aa8a69152f413524054f0b5980c28db1e07c83`.

The rebuilt ZIP was expanded into a new GUID-named empty directory under the
system temporary directory. The validator script from that extraction was run
against the rebuilt ZIP and extracted `submission/PA2_report_final.pdf`. It
exited **0** and reported counts 73/105/128, `bad_hashes: []`, zero duplicate
entries, no missing or forbidden entries, no hygiene findings, exactly one
report PDF, all eight intended LoRA weights, and no missing manifest files.

## HH beta generation integration — 17 October 2026

The completed 300-prompt generation checks and both blinded judge runs were
analyzed and packaged as 38 additional hash-manifested evidence files. The
report-number checker passed with beta generation checking enabled. The report
remains eight pages and its three copies are byte-identical. The unchanged
public suite passed **7/7** in 10.57 seconds; the complete suite passed
**77/77** in 41.05 seconds with the same 14 third-party warnings. The local
archive validator passed with 533 unique files and evidence counts
73/105/128/38, for 344 hash-manifested artifacts total.

## Figure, adapter-edge, and hygiene pass — 17 October 2026

The report was regenerated from JSON and rendered at 110 dpi page by page. Figure 1 now uses full-width SFT, HH, and UltraFeedback rows with external DPO legends and selected-update markers; Figure 2 uses distinct colors and markers, a log loss axis, data-derived limits, and an external legend. The generated training text records the actual FP32-policy/FP16-reference, cuda:0/cuda:1, FP16-autocast cause of the tiny initial DPO margins.

Commands executed with Python 3.12.10 and torch 2.14.1+cpu:

```powershell
$env:PYTHONPATH='.'
.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_public_adapter_smoke.py 2>&1 | Tee-Object docs\public_pytest_output_2026-10-17.txt
.venv-local\Scripts\python.exe -m pytest -q -p no:cacheprovider 2>&1 | Tee-Object docs\pytest_output_2026-10-17.txt
```

Observed results: unchanged public tests **7/7** in 13.96 seconds; complete suite **83/83** in 91.61 seconds with 14 third-party Matplotlib/PyParsing deprecation warnings. The raw command output is retained in the two dated text files. `scripts/check_report_numbers.py` and compilation also passed.

## Authentic executed setup notebook — 17 October 2026

Kaggle kernel `moibbi/pa2-executed-environment-notebook` v5 completed. Its downloaded `00_setup_and_env_check.executed.ipynb` was verified before copying: the first cell is the Evidence markdown cell; three output objects are saved; `nvidia-smi` is present; `torch.cuda.device_count()` is 2; both device names are Tesla T4; and torch, transformers, datasets, and accelerate versions are printed. The notebook bytes were copied unchanged to `notebooks/00_setup_and_env_check.ipynb`; no output was synthesized or edited.

Final publication gate used `PYTHONWARNINGS=ignore`: public tests **7/7** in
9.52 seconds and complete suite **83/83** in 38.61 seconds. These
results are the contents of the dated raw pytest output files packaged with the
submission; PowerShell's UTF-16 capture was decoded and rewritten losslessly as
UTF-8 for archive portability.

## Final diagnostic-figure correction — 17 October 2026

The four report figures were regenerated from the saved metrics at the exact
LaTeX text width, with 2.3/2.5/2.5/2.8-inch heights, data-derived padded limits,
external legends, and explicit selected-update markers. Three added regression
tests verify PDF dimensions, plotted-data containment, and the required
accuracy/persona extrema. The unchanged public suite passed **7/7** in 16.61
seconds and the complete suite passed **86/86** in 67.73 seconds with warnings
suppressed for the retained portable logs. The report-number checker passed.
Three `pdflatex -interaction=nonstopmode -halt-on-error` passes produced an
eight-page report with no overfull/underfull boxes, undefined references,
missing glyphs, or LaTeX warnings. `latexmk` was attempted first but could not
start because this MiKTeX installation has no Perl script engine; no document
compilation occurred in that failed attempt. Every page from the successful
build was rendered at 100 dpi and inspected.
