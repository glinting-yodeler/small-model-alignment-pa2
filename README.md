# Small-Model Alignment Starter

This repository now includes the PA2 implementation. The original starter
instructions below are retained for provenance; current execution status and
reproduction commands follow them.

**Edge cases.** EOS is appended before truncation, which may remove it. The
first response token after an empty tokenized prompt is unscored because it has
no causal context, and training emits one aggregate warning for such prompts.
A prompt filling the length cap yields a finite zero-token row. PAD equal to
EOS is separated by attention/label masks; missing EOS appends nothing; mixed
lengths and DPO rows preserve input order.

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.

## Local implementation added

The original starter text above is preserved. The six adapter functions now use
the reusable `alignment` package. Main SFT and both DPO tracks use custom PyTorch
loops; only persona uses TRL/PEFT. Configs, scripts, the authentically executed
setup notebook, remaining reproducibility notebooks, and
offline tests cover preparation, training/resume, generation, blinded pairs,
Kaggle judging and plots.

See [reproducibility and commands](docs/reproducibility.md),
[recorded preflight and gated-pilot evidence](docs/experiment_log.md),
[local test evidence](docs/local_validation.md), and
[changed-file inventory](docs/file_changes.md). Saved Kaggle run links and
downloaded evidence are in [artifact links](docs/artifact_links.md).

The CPU CLI smoke and integrated T4×2 pilot passed. Frozen preparation produced
30,000/1,000 SFT, 15,000/1,000 HH, 15,000/1,000 UltraFeedback, and 300
AlpacaEval records. Full SFT, persona, HH DPO, and UltraFeedback DPO training
completed on Kaggle. Fixed-set generation and all four blinded evaluation tracks
also completed; the measured report and evidence are under `submission/`.
Current run IDs, downloaded evidence, and failures/resumes are recorded
in [the experiment log](docs/experiment_log.md). The unchanged public suite
passes 7/7. The expanded local suite currently has 58 passing tests and one
intentional provenance-check conflict documented in `docs/local_validation.md`:
the assignment requires adding TRL/PEFT to `requirements.txt`, while that check
expects the untouched starter-file hash.

Run offline tests with `python -m pytest -q` in an environment with
`requirements.txt` installed. Rebuild a Kaggle package with
`python scripts/prepare_kaggle_stage.py sft|dpo_hh|dpo_ultrafeedback|persona`
or `python scripts/prepare_kaggle_generation.py sft|hh|ultrafeedback|persona`.
Generated packages are local staging files; the source configs and preparation
scripts are tracked. `docs/reproducibility.md` gives the exact data, training,
generation, pairing, judging, plotting, and report commands.

The task notebooks are `notebooks/00_setup_and_env_check.ipynb` through
`notebooks/06_judge.ipynb`. `scripts/env_check.py` records GPU and library
versions; `scripts/sanity_check_generation.py` audits 2–5 outputs before full
generation; and `scripts/analyze_results.py` rebuilds quantitative error
analysis from saved artifacts. Persona v2 and optional HH beta-pilot configs are
prepared but have no claimed results. See `START_HERE.md` for the exact remaining

## Final manual-compliance status

The submitted 300-row generations and all saved pairs/judgments remain unchanged.
No preserved record demonstrates that the required 2–5 prompt check preceded
those full runs. A labelled two-prompt post-run verification is saved under
`outputs/analysis/`; it used the submitted persona adapter and does not replace
historical output. The response-length threshold in analysis is a re-tokenized
saved-text proxy, not a confirmed generation-cap hit.

Repository: https://github.com/glinting-yodeler/small-model-alignment-pa2

Persona v3 is a preregistered follow-up to the reported v2 degeneration. The
historical selection rule and the revised held-out-generation rule are recorded
in `docs/persona_selection_rule_v2.md`; neither uses final judge outcomes for
selection. Configurations `persona_lora_v3a.yaml` through `v3d.yaml` and their
generation configs are packaged by `python scripts/prepare_kaggle_stage.py
persona_v3`. Apply the revised rule with `scripts/select_persona_adapter.py`
after downloading the kernel outputs.

Persona v3 completed on Kaggle. The revised held-out rule selected `v3b` before judging. Its blinded results against base are style 7/7/6 (50.0%, 95% CI 32.5--67.5) and quality 5/10/5 (37.5%, 20.0--57.5). Additive artifacts are under `submission/evidence_v3/`.
