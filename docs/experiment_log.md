# Experiment log

## 12 October 2026 — local implementation

- User-reported CLI CPU smoke passed; existing downloaded JSON checked.
- User-reported T4x2 smoke remains queued. No new remote jobs submitted.
- Starter preserved before implementation; source changes and offline tests only.
- External dataset schema samples and source revisions inspected. No full dataset
  subset preparation or pretrained model training was run locally.
- Tiny random CPU models exercise custom loops, TRL/LoRA, reload and generation.
  These tests do not measure assignment model quality.
- At that historical checkpoint, full Kaggle pilot/training, GPU timing,
  model-based judgments, evaluation metrics, and report results were pending;
  the later entries below supersede this status.

For each subsequent authorized stage append: date/time, source/config/data
hashes, kernel/version/output artifact, accelerator evidence, measured wall time
and memory, actual update counts, result/failure and any justified deviation.

## 13 October 2026 — completed and active Kaggle stages

- Manual CPU CLI round trip succeeded. The original queued T4 smoke was
  superseded by the integrated `moibbi/pa2-small-gpu-pilot` version 3, which
  completed on two Tesla T4 devices. The downloaded pilot result reports 496
  seconds, SFT/DPO checkpoint reloads, unchanged frozen DPO references, and a
  changed/reloaded persona adapter. Earlier pilot failures and fixes are recorded
  below with the corresponding kernel versions and downloaded artifacts.
- Frozen private input dataset `moibbi/pa2-frozen-prepared-inputs` contains the
  prepared 30,000/1,000 SFT, 15,000/1,000 HH, 15,000/1,000 UltraFeedback,
  and 300 AlpacaEval records. Its file hashes are in its upload manifest.
- `moibbi/pa2-sft-full` version 3 completed. Its downloaded manifest identity
  is `c0eb6e51f6de1d5c8c82b803a9eb029868de4aeab045f823f88e7da71a60eb0c`;
  completion records 938 optimizer steps, 6147.25 seconds, best validation loss
  1.372005, and 10,606,173,696 peak CUDA bytes on GPU 0. The large local best
  model download failed twice due interrupted HTTP reads; its Kaggle artifact
  remains attachable and the DPO jobs are using it.
  A resumable byte-range transfer subsequently downloaded the complete
  1,976,163,472-byte SFT best weight file. The local tokenizer/model reload
  succeeded and produced finite logits with shape `(1, 3, 151936)` on a probe.
  Its local SHA-256 is
  `5f34751f09f214d0986d8ae7bcb96b1021c08256d206214da2dfc601a00c3545`.
  The saved validation trace is 1.4030 at step 0, peaks at 1.4283 after step
  100, then decreases to 1.3720 at step 938; this describes the measured trace
  without claiming a monotonic gain.
- The integrated pilot's three saved SFT generations were downloaded and
  inspected before the 300-prompt full generation job. All three decode as
  response text with the expected prompt IDs and 48-token pilot cap. At least
  one response (the extinct-bear prompt) contains factual errors, which should
  be treated as a model-quality limitation, not a pipeline failure.
- A second three-prompt CPU sanity check loaded the downloaded *full SFT best*
  checkpoint and completed response-only generation under the same template
  with a 96-token local cap. The outputs in `outputs/sft-final-sanity/` preserve
  the expected IDs and are coherent as text, but the extinct-bear response again
  lists living bears as extinct. This validates generation plumbing while
  confirming a visible quality error before the 300-prompt comparison.
- `moibbi/pa2-persona-full` version 2 completed on 198/22 preference pairs.
  Its adapter was reloaded and its measured validation loss is about 0.648862.
  The 3,293,480-byte adapter weight file and tokenizer are now downloaded
  locally; the safetensors file opens with 128 LoRA tensors. Its local SHA-256
  is `fb832c106783b25d99ba94b184af11c01461df8e0378edcc85652e398fc0444f`.
  `moibbi/pa2-generate-persona` version 1 saved aligned base/LoRA generations
  for 20 fixed prompts; five of 20 response texts differ.
- Two private benchmark tasks used saved randomized persona pairs and
  `google/gemini-3.6-flash` actual judge calls. Persona style version 1 parsed
  only five decisions because valid JSON responses were fenced; all raw attempts
  were saved. A strict outer-fence parser fix and attached resume journal yielded
  style version 2 with 20/20 coverage: 0 wins, 1 loss, 19 ties. Persona quality
  version 1 yielded 20/20: 1 win, 3 losses, 16 ties. Both judgment rows and
  summaries are downloaded and staged under `outputs/evaluation/`.
- `moibbi/pa2-dpo-hh-full` and `moibbi/pa2-dpo-ultrafeedback-full` version 1
  completed from the same SFT best checkpoint, whose recorded SHA-256 is
  `5f34751f09f214d0986d8ae7bcb96b1021c08256d206214da2dfc601a00c3545`
  in both resolved configs. Each track completed 469 optimizer updates. HH took
  9009.28 seconds and selected validation reward accuracy 0.592 with margin
  0.17554; UltraFeedback took 8429.78 seconds and selected validation reward
  accuracy 0.701 with margin 0.51148. Both used two Tesla T4 devices and peaked
  near 10.74 GB on the policy GPU. Fixed-set SFT and HH generation version 1
  were then pushed; SFT is running and HH is queued. UltraFeedback generation
  is prepared and awaits a free Kaggle GPU slot.
- `moibbi/pa2-generate-sft` version 1 completed three aligned 300-row sets:
  base AlpacaEval, SFT AlpacaEval, and SFT HH. `moibbi/pa2-generate-hh`
  version 1 completed 300 HH DPO outputs. All sets have unique IDs, nonempty HH
  responses, checkpoint-bound manifests, and completion hashes. HH DPO changes
  280/300 decoded responses relative to SFT; SFT changes 295/300 AlpacaEval
  responses relative to base.
- The immutable 300-row HH blinded set was uploaded privately as
  `moibbi/pa2-hh-blinded-pairs`. Benchmark task `moibbi/pa2-judge-hh` version 1
  completed with 300/300 valid decisions from the recorded
  `google/gemini-3.6-flash` judge: 126 DPO wins, 90 losses, and 84 ties. The
  decisive win rate is 0.58333 and tie-adjusted score is 0.56; the locally
  recomputed summary exactly matches the downloaded summary.
- `moibbi/pa2-generate-ultrafeedback` version 1 completed 300/300 unique,
  nonempty outputs with an exact completion-hash match. It changes 290/300
  decoded AlpacaEval responses relative to SFT. The immutable blinded set is
  private dataset `moibbi/pa2-ultrafeedback-blinded-pairs`; benchmark task
  `moibbi/pa2-judge-ultrafeedback` version 1 completed with 300/300 valid
  decisions from the recorded `google/gemini-3.6-flash` judge: 86 DPO wins,
  132 losses, and 82 ties. The decisive win rate is 0.39450 and tie-adjusted
  score is 0.42333, with no unknowns or failures. The locally recomputed
  summary exactly matches the downloaded summary.

## Persona v2 model-selection rule (pre-declared 16 October 2026)

Before inspecting any v2 judge results, the selection rule is: choose the
variant with the highest held-out reward accuracy; break ties using larger
held-out reward margin, then lower evaluation loss. Report v1, v2 at 2e-4, and
v2 at 1e-4 together. Blinded judge scores are evaluation results and are not a
selection criterion.

- 2026-10-16T01:52:25.3421305+05:00 - kaggle kernels push -p kaggle_smoke/gpu: moibbi/pa2-cli-gpu-smoke-oct12 version 5 COMPLETE; output downloaded to outputs/kaggle_preflight/gpu_smoke.

- 2026-10-16T01:53:13.4960110+05:00 - persona v2 push requested metadata slug moibbi/pa2-persona-v2-sweep; Kaggle created canonical slug moibbi/pa2-persona-v2-full version 1. Status RUNNING. Updated local builders to use the canonical source slug.

- 2026-10-16T02:15:06.1152663+05:00 - moibbi/pa2-persona-v2-full version 1 COMPLETE. Downloaded outputs/persona_v2. Verified v1=12 and both v2 variants=72 optimizer updates, finite adapter tensors, reload ok, and held-out metrics present. Predeclared rule selects persona_v2 LR 2e-4 (accuracy tie at 1.0; margin 8.9179 vs 4.9076 vs 0.0907).

- 2026-10-16T02:20:20.4868826+05:00 - moibbi/pa2-generate-persona-v2 version 1 ERROR in sanity check: vars(BatchEncoding) passed internal data/_encodings/_n_sequences to generate. Fixed sanity_check_generation.py to pass dict(inputs); focused offline regression passed; repushing.

- 2026-10-16T13:07:36.6855587+05:00 - pushed moibbi/pa2-dpo-hh-pilots-full v1 (three 150-update matched beta pilots) and moibbi/pa2-env-t4x2-check v1 concurrently; both RUNNING.

- 2026-10-16T13:10:19.5082122+05:00 - moibbi/pa2-env-t4x2-check v1 ERROR: auxiliary requirements file was not included by Kaggle script upload. Embedded requirements and env_check source into run.py; repushing v2.

- 2026-10-16T13:13:57.3011792+05:00 - moibbi/pa2-env-t4x2-check v2 COMPLETE: 2x Tesla T4; torch 2.11.0+cu128, transformers 4.51.3, datasets 3.6.0, TRL 0.18.1, PEFT 0.15.2, accelerate 1.6.0. Outputs saved under outputs/analysis/env_kaggle. Pushing v3 as notebook solely to preserve genuine executed notebook outputs.

## 2026-10-16 final v2 integration

- `moibbi/pa2-generate-persona-v2` v1 ERROR: the pre-generation sanity script expanded internal `BatchEncoding` fields into `generate`. Version 2 fixed this with `dict(inputs)` and completed both variants. Each has 20/20 unique IDs, a distinct generator name, adapter hash, finite/reload-verified source adapter, and a saved three-prompt sanity check produced before the full run.
- Persona benchmark tasks completed with zero unknowns/failures: `pa2-judge-persona-v2-style` v2 (run 4802960), `pa2-judge-persona-v2-quality` v1 (4802966), `pa2-judge-persona-v2-lr1e4-style` v1 (4802970), and `pa2-judge-persona-v2-lr1e4-quality` v1 (4802972). Kaggle displayed the runner as Gemini 3.8 Flash; every task manifest and judgment row records the actual fixed judge `google/gemini-3.6-flash`. Calls used batches of five, at least five seconds between attempts, and at most three attempts.
- The predeclared selection rule chose v2 LR 2e-4 from held-out metrics, before judgments. Its style and quality results were both 0W/20L/0T. LR 1e-4 produced 3W/17L/0T style and 3W/16L/1T quality. These results do not alter selection.
- `moibbi/pa2-env-t4x2-check` v3 COMPLETE as a notebook kernel. Kaggle CLI pull returned source without cell outputs; genuine downloaded v2 JSON and `nvidia-smi.txt` are preserved, and no notebook outputs were synthesized.
- Exact prepared-data scan: 64,000 SFT/HH/UltraFeedback train+validation rows; zero empty source prompts and zero empty formatted strings. Runtime encoding also counts and warns if tokenization yields an empty formatted prompt. Evidence: `outputs/analysis/empty_prompt_audit.json`.
- Report data were regenerated with `scripts/build_report_tex.py`; `scripts/check_report_numbers.py` checks generated macros and persona/beta table entries against JSON.
- 2026-10-16T15:26:00+05:00 - `moibbi/pa2-dpo-hh-pilots-full` v1 ERROR after beta 0.05 completed and beta 0.1 reached a checkpoint save: Kaggle working storage failed while serializing a multi-gigabyte optimizer state (`basic_ios::clear`). Downloaded logs to `outputs/hh_beta_pilots/kernel.log`. Validation-only pilots do not need model exports, so `metrics_only` now retains metric journals/resolved configs/completion metadata while skipping best/final/latest model files. Normal checkpoint/resume tests passed. Initial repush hit a canonical-slug metadata conflict; corrected the builder from `pa2-dpo-hh-beta-pilots` to the existing `pa2-dpo-hh-pilots-full`. Version 2 pushed successfully and is RUNNING.
- 2026-10-16T18:10:00+05:00 - `moibbi/pa2-dpo-hh-pilots-full` v2 COMPLETE after the metrics-only disk fix. Downloaded three resolved configs, manifests, completions, and complete metric journals to `outputs/hh_beta_pilots_v2/`. Update-150 validation results: beta 0.05 accuracy 0.588, margin 0.097350, loss 0.674708; beta 0.1 accuracy 0.589, margin 0.134535, loss 0.675904; beta 0.3 accuracy 0.578, margin 0.185100, loss 0.693592. The control has the highest final accuracy by a narrow margin; this is validation-only evidence and did not replace or regenerate the submitted HH model.
- 2026-10-16T18:24:51.2584657+05:00 - Declared the revised persona adapter selection rule in docs/persona_selection_rule_v2.md before any persona-v3 generation or judgment. The rule uses only the 22 held-out validation prompts; final evaluation prompts and judge outputs are excluded from selection.

- 2026-10-16T18:28:40.8718859+05:00 - Pushed `moibbi/pa2-persona-v3-full` version 1 with `kaggle kernels push -p kaggle_stages/persona_v3`. Expected outputs: environment JSON; v3a-d metrics, adapters, finite/reload checks; pre-generation sanity records; 22-prompt base plus seven-adapter held-out generations; v3a-d 20-prompt evaluation generations; summary JSON. Planned local download: `outputs/persona_v3/`. Status at push: submitted; deliberately not polled under the three-session protocol.

- 2026-10-16T21:50:35.5370190+05:00 - Applied the predeclared revised persona rule to the 22 held-out prompts before any v3 judging. Selected `v3b`: base degeneration 0.0%; v3b degeneration 4.55%, changed 100%, style-marker rate 63.64%. v1 was also eligible (0.0%, 40.91%, 59.09%); all other variants failed the degeneration threshold. Judge results were not available or used.

- 2026-10-16T21:55:10.6136943+05:00 - Pushed Kaggle Benchmark task `pa2-judge-persona-v3` with private dataset `moibbi/pa2-persona-v3-pairs` (dataset v2 contains all eight 20-pair files). The task covers v3a-d x style/quality, fixed judge `google/gemini-3.6-flash`, batches of five, >=5 s delay, at most three attempts, and resumable journals. The server returned the new creation state `VALIDATING`; Kaggle CLI 2.2.4/kagglesdk 0.1.37 created the request but crashed while deserializing that unknown enum. Added the missing enum only to the local CLI environment so the next session can query status; task source and evaluation code were not changed.

- 2026-10-16T23:43:00.9139822+05:00 - Kaggle Benchmark task `pa2-judge-persona-v3` v1 COMPLETE, run 4809623. Downloaded all eight style/quality tracks. Each has 20/20 coverage and zero unknown/failure; actual judge manifests record `google/gemini-3.6-flash`, runner display is Gemini 3.8 Flash. Minimum saved inter-attempt interval was 5.00016 seconds. Revised-selected v3b: style 7/7/6, score 0.500 CI [0.325,0.675]; quality 5/10/5, score 0.375 CI [0.200,0.575].

## 17 October 2026 — HH beta generation check

- `moibbi/pa2-hh-beta-generation` v1 completed the matched 150-update beta
  reproductions and immediately generated the fixed 300 HH prompts from each
  best checkpoint. All three sets have 300 unique, nonempty outputs and hashes
  matching the saved kernel summary. Temporary full checkpoints were deleted
  after generation.
- Blinded comparisons used beta 0.10 as the reference. Benchmark run 4888390
  judged beta 0.05 at 67W/56L/177T, score 0.5183, bootstrap 95% CI
  [0.4833,0.5550]. Run 4888391 judged beta 0.30 at 77W/71L/152T, score 0.5100,
  CI [0.4717,0.5500]. Both had 300/300 coverage and zero unknowns/failures.
  Neither interval excludes 50%, so the generation check supplies no conclusive
  reason to replace the submitted beta 0.10 setting.

- 2026-10-17 - `moibbi/pa2-executed-environment-notebook` v5 COMPLETE. Downloaded and verified the executed setup notebook: saved outputs include `nvidia-smi`, CUDA device count 2, two Tesla T4 names, and required package versions. Copied its bytes unchanged into notebook 00.
