# PA2 submission handoff

Repository: https://github.com/glinting-yodeler/small-model-alignment-pa2

The final deliverables are `PA2_Submission.zip` and the separate
`PA2_Report.pdf` placed next to it in the ready-to-submit folder. The archive's
`report/` directory contains the LaTeX sources and figures; generated report
PDFs are intentionally excluded. The preserved v1 evidence is under
`submission/evidence/`; additive persona v2, environment, and analysis evidence
is under `submission/evidence_v2/`; the completed HH beta generation check is
under `submission/evidence_beta/`.

Persona v3 is currently an additive follow-up. Its predeclared revised selection
rule is `docs/persona_selection_rule_v2.md`; downloaded artifacts will be stored
under `outputs/persona_v3/` and packaged without changing evidence v1 or v2.

The custom PyTorch SFT and main DPO loops are in `alignment/`. Persona alone uses TRL/PEFT. Reproducible commands and exact Kaggle artifacts are in `docs/reproducibility.md`, `docs/experiment_log.md`, and `docs/artifact_links.md`.

The seven public tests remain unchanged. Run the complete offline suite with `.venv-local\Scripts\python.exe -m pytest -q`. Rebuild the report inputs with `.venv-local\Scripts\python.exe scripts/build_report_tex.py`, verify numbers with `.venv-local\Scripts\python.exe scripts/check_report_numbers.py`, and compile `report/main.tex`.

The verified public repository is published on the `main` branch. Use `git
rev-parse HEAD` and `git ls-remote origin refs/heads/main` to confirm the exact
submitted revision before uploading the deliverables.

The final additive persona-v3 evidence is in `submission/evidence_v3/`. The revised rule selected v3b before judgment; all seven variants and both historical/revised choices are reported.
