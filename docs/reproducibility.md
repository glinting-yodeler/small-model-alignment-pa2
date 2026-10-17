# Local implementation and experiment workflow

GitHub repository: https://github.com/glinting-yodeler/small-model-alignment-pa2

The implementation and measured Kaggle stages are recorded here. The
24-page assignment PDF and complete handoff plan were read. The original starter
is preserved in `../pa2-starter-preserved-20261012.zip` (19 files, excluding Git
internals and environments), with SHA-256 hashes in `docs/starter_manifest.json`.
The preexisting plan, smoke scripts, downloaded CPU evidence, public tests,
assignment PDF, supplied persona data and HH evaluation data are unchanged.

## Edge cases

EOS is appended before truncation, so truncation may remove it. The first
response token for an empty tokenized prompt is deliberately unscored because
causal shifting provides no preceding context; training counts these rows and
emits one aggregate warning. A prompt that fills `max_length` produces a finite
scalar zero-token loss. PAD equal to EOS remains safe because attention and
label masks identify padding, while a tokenizer without EOS appends no terminal
token. Mixed-length rows and chosen/rejected DPO rows preserve input order.

## Environment and tests

The default `python` on this machine is MSYS Python without torch. Use the
separate Windows CPython 3.12 environment `.venv-local`, not `.venv-kaggle`:

```powershell
py -3.12 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-local\Scripts\python.exe -m pip install -r requirements-kaggle.txt
.\.venv-local\Scripts\python.exe -m pip install -e . --no-deps
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
.\.venv-local\Scripts\python.exe -m pytest -q
```

The torch installation line is **local CPU only**. Keep Kaggle's installed CUDA
torch. The pilot installer constrains torch to its already-installed version.
`requirements.txt` is preserved; `requirements-kaggle.txt` supplies the
versioned Transformers/Datasets/TRL/PEFT stack used in the completed T4 pilot
and full SFT/persona jobs. Optional torchao conflicts should be diagnosed from
actual errors.

Tests include real tiny randomly initialized Transformers models and a tiny
TRL/PEFT optimization/reload, generated entirely offline. They are correctness
tests, not model quality or Kaggle throughput measurements.

## Core conventions

- Adapter prompt/response strings are encoded separately with
  `add_special_tokens=False`, then concatenated. No template/BOS is inserted.
  An EOS is appended whenever defined, before total-length truncation, even if
  the caller supplied a terminal EOS already. This explicit boundary convention
  follows the adapter contract; upstream plain-text datasets have no inserted EOS.
- Right padding is determined by actual lengths; PAD=EOS retains genuine EOS
  targets. Missing PAD falls back to EOS, then vocabulary index 0, without
  mutating the tokenizer. Empty batches/mismatched lengths are errors.
- Logits at t predict labels at t+1. Scores are response sums, not length means.
  An unconditioned first token cannot be scored. Empty masks return zero scores
  and a differentiable zero SFT loss; training rejects such examples.
- SFT uses total response-token NLL / total shifted response-token count.
  Accumulation and validation use exact token totals, including partial windows.
- DPO uses stable `-logsigmoid(beta * (policy_gap-reference_gap))`, pair means,
  frozen eval/no-grad reference, and strict reward accuracy (initial exact ties=0).
  Policy remains differentiable. Reference scores move to the policy device.
- Training formats prompts as `### Instruction:\n{instruction}\n\n### Response:\n`.
  DPO retains at most the final 256 prompt tokens within 512 total; the generic
  adapter does not apply this training-only cap. Generation uses the full prompt
  and fails on context overflow, rather than silently dropping instruction text.
- SFT and both main DPO tracks use only the custom `alignment/training.py` loop.
  Main policy weights are FP32 with FP16 autocast/scaling on CUDA. DPO reference
  is FP16 on cuda:1, policy on cuda:0; this is **not** data parallel training.
  Finite-precision initial DPO rewards may differ slightly from exact toy ties.
- AMP overflow retries the same window; scheduler advances only after a
  successful update. DPO uses fixed LR. Best SFT minimizes validation loss;
  best DPO maximizes reward accuracy, then minimizes loss, retaining earlier ties.

## Data preparation

Run only in a network-enabled preparation environment (remove offline env vars
for this stage). No model weights are needed for data preparation:

```bash
python scripts/prepare_data.py --config configs/data.yaml
```

The config pins model/tokenizer and dataset revisions. Sources and inspected schemas:

| Track | Dataset/split | Parsing |
|---|---|---|
| SFT | HuggingFaceH4/ultrachat_200k, train_sft | first user/assistant exchange; optional system text retained |
| HH | Anthropic/hh-rlhf, four named train files | strict shared single Human/Assistant exchange; red-team-attempts excluded |
| UF | HuggingFaceH4/ultrafeedback_binarized, train_prefs | shared conversational prefix and distinct assistant alternatives |
| Alpaca | tatsu-lab/alpaca_eval, alpaca_eval.json | 300 source indices sampled with Python Random(42), saved in order |

Alpaca's pinned JSON is loaded directly, avoiding execution of its dataset script.
The file has 805 records. Source cards list MIT for the three training datasets,
CC-BY-NC-4.0 for AlpacaEval, and Apache-2.0 for both base models. Cards/metadata
were checked. The full preparation run completed; source revisions, exact
subsets, rejection counters, hashes,
license metadata and seed are written with every prepared dataset.
The lightweight ZIP packages the Alpaca evaluation source indices and the
training source revisions, filtering counts, and selected-subset hashes in
`submission/evidence/data/`. Individual selected training source indices and
records are stored in the full prepared `*_train.json` and
`*_validation.json` datasets, which are outside the lightweight ZIP. A subset
hash does not reveal which individual source rows were selected.

Malformed/empty/identical pairs, multi-turn preferences, normalized duplicate
prompts, fixed evaluation overlaps and unscoreable truncated examples are filtered
before selection. One record is retained per normalized prompt. Train/validation
are prompt-disjoint. Insufficient usable counts fail explicitly. Full preparation
scans the pinned source; the pilot scans at most 5,000 rows per source and requests
16 train/4 validation rows. Full configs enforce 30,000/1,000 SFT and 15,000/1,000
for each DPO track. The frozen prepared counts were measured as 30,000/1,000
SFT, 15,000/1,000 HH, 15,000/1,000 UltraFeedback, and 300 AlpacaEval prompts.

The supplied HH set has 300 unique IDs but 185 unique prompt texts. Preserve all
300 evaluation rows, including repeated text. The supplied persona set has 220
unique prompts; the separate frozen `persona_eval_20.json` has five prompts each
for cooking, coding, general help and safety, checked for train overlap.

## Full training commands and completed stages

From the checkout in Kaggle, after data preparation and a successful pilot:

```bash
python scripts/train_sft.py --config configs/sft.yaml
python scripts/train_dpo.py --config configs/dpo_hh.yaml
python scripts/train_dpo.py --config configs/dpo_ultrafeedback.yaml
python scripts/train_persona.py --config configs/persona_lora.yaml
```

Use separate saved Kaggle jobs/stages. Both DPO configs point to the same
`/kaggle/working/outputs/sft/best`; attach or copy that preserved artifact before
each stage. A checkpoint marker requires track=sft and kind=best; weight hashes
are recorded. Never replace this with HH DPO or the original Qwen checkpoint.
Full effective batch is 32 for SFT (2x16) and each DPO (1x32). Defaults give 938
and 469 updates respectively for one epoch with the final partial window.
The completed SFT kernel is `moibbi/pa2-sft-full` version 3; persona is
`moibbi/pa2-persona-full` version 2. Both independent DPO kernels completed
version 1 after SFT. Their downloaded configs record the same SFT checkpoint
hash, and their measured outcomes are recorded in `docs/experiment_log.md`.

Persona uses SmolLM2's chat template via TRL exactly once, rank 8, alpha 16,
dropout .05, q_proj/v_proj (validated against actual module names), LR 5e-5,
beta .1, microbatch 1 x accumulation 16, 198/22 prompt-disjoint rows and one
epoch. These are proposed choices, not measured optimal settings. The script
limits visible GPUs to one unless the caller already set CUDA_VISIBLE_DEVICES.
TRL's disabled-adapter context supplies the original base reference, verified at
runtime. It saves adapter-only weights and a pinned base revision, then checks
reloaded logits. Persona quality and style are judged separately.

## Checkpoints and recovery

Each main run saves `best/`, `final/`, `latest.pt`, `metrics.jsonl`,
`resolved_config.json`, `manifest.json`, `completion.json`. Latest state includes
model, optimizer, scheduler, scaler, sampler epoch/offset, best-selection state,
step/attempt counters and Python/CPU/CUDA RNG. It is saved only at optimizer
boundaries. Resume requires identical config/data/source identity:

```bash
python scripts/train_sft.py --config configs/sft.yaml --resume /kaggle/working/outputs/sft/latest.pt
```

Use the same output path (copy attached output to writable storage if necessary).
Weight loading alone is not exact resume. Only load trusted training-state files.
Do not edit code/config between an exact resume; make a new run for changed
experiments. Metrics are append-only; after a crash between journal writes and
the last state save, use the checkpoint step to distinguish any replayed log
steps. CPU deterministic resume is tested. The integrated GPU pilot verified
reload and frozen-reference behavior; interrupted full-run exactness has not
been exercised on Kaggle.

There is one bounded latest optimizer state plus best/final inference checkpoints.
Full fine-tuning artifacts are large. Preserve each stage using Save Version or
an output dataset, confirm it can be attached/read, and avoid accumulating all
optimizer states in one job. An archive on the same ephemeral disk is not an
independent backup. The pilot removes only its verified redundant final/latest
weights after auditing reload; it retains best checkpoints and evidence.

## Generation and blinded evaluation

There are seven ready generation configs:
`generate_base_alpaca`, `generate_sft_alpaca`, `generate_ultrafeedback_alpaca`,
`generate_sft_hh`, `generate_dpo_hh`, `generate_persona_base`,
`generate_persona_lora` (all `.yaml` in configs). Run each with:

```bash
python scripts/generate.py --config configs/generate_sft_alpaca.yaml
```

Generation uses greedy decoding, left padding, response-only decoding, a 256
new-token budget, and repetition penalty 1.0. Temperature/top-p are inactive.
Inspect 2–5 pilot responses before full generation. The JSONL journal resumes
by ID only with matching checkpoint, source, prompts and decoding identity; the
completed JSON preserves original order. Keep comparison decoding identical.

```bash
python scripts/build_pairs.py --baseline /kaggle/working/outputs/generations/sft_hh.json --candidate /kaggle/working/outputs/generations/dpo_hh.json --track hh --output /kaggle/working/outputs/eval/hh_pairs.jsonl
python scripts/build_pairs.py --baseline /kaggle/working/outputs/generations/sft_alpaca.json --candidate /kaggle/working/outputs/generations/ultrafeedback_alpaca.json --track ultrafeedback --output /kaggle/working/outputs/eval/ultrafeedback_pairs.jsonl
python scripts/build_pairs.py --baseline /kaggle/working/outputs/generations/persona_base.json --candidate /kaggle/working/outputs/generations/persona_lora.json --track persona_style --count 20 --output /kaggle/working/outputs/eval/persona_style_pairs.jsonl
python scripts/build_pairs.py --baseline /kaggle/working/outputs/generations/persona_base.json --candidate /kaggle/working/outputs/generations/persona_lora.json --track persona_quality --count 20 --output /kaggle/working/outputs/eval/persona_quality_pairs.jsonl
```

Pair construction joins IDs and validates prompt/set/count, randomizes order
with seed 42, and stores private mappings and content hashes. Model labels never
enter judge requests. Candidate text remains untrusted data in the judge rubric.

Use a **separate utility benchmark notebook** for judge access. The backend uses
the documented `kbench.llms[model]`, fresh chat context, `llm.prompt` API. The PDF's
`google/gemini-3.6-flash` was verified by a separate private benchmark access
check and used for the saved persona judgments. The benchmark task wrapper was
run by Gemini 3.7 Flash, but every saved decision row records the actual Gemini
3.6 Flash judge. The code never silently substitutes a model.

```bash
python scripts/judge_pairwise_kaggle.py --pairs-path /kaggle/input/pa2-pairs/hh_pairs.jsonl --output-path /kaggle/working/hh_judged.jsonl --judge-model google/gemini-3.6-flash --rubric hh --batch-size 5 --sleep-seconds 5 --max-attempts 3
python scripts/summarize_results.py --pairs /kaggle/input/pa2-pairs/hh_pairs.jsonl --judgments /kaggle/working/hh_judged.jsonl --output /kaggle/working/hh_summary.json
```

Use `general` for UltraFeedback. Persona style and quality use separate saved
pair files with identical prompts/responses but independent comparison IDs and
rubrics. The journal always resumes;
completed compatible results are not requested again. Three total attempts per
batch **per invocation**, not three retries; unresolved pairs can receive a new
attempt budget on a later invocation. All request attempts, including retries
and interrupted calls, are journaled before calling and separated by >=5 seconds.
Partial valid results are retained; only unresolved pairs are retried. Malformed
responses never imply a winner. An explicit unknown is a completed judgment.

Summary denominators: W/(W+L), (W+.5T)/(W+L+T), W/(W+L+T), valid coverage
(W+L+T)/N, U/N, F/N. Zero denominators are null. N=W+L+T+U+F. `retries` counts
extra attempts for completed pairs; `pair_retries_including_failures` and
`request_attempts` use the attempt journal. Tracks are never pooled.

The judge CLI accepts both hyphenated and underscored flag spellings. The
manual-style command below resumes compatible saved judgments; the explicit
flag documents that intent. The minimum five-second delay is still enforced.

```bash
python scripts/judge_pairwise_kaggle.py --pairs_path /kaggle/input/pa2-pairs/hh_pairs.jsonl --output_path /kaggle/working/hh_judged.jsonl --judge_model google/gemini-3.6-flash --rubric hh --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

Before a new full generation, run 2–5 prompts and inspect the saved record:

```bash
python scripts/sanity_check_generation.py --config configs/generate_persona_lora_v2.yaml --count 3
```

The v1 Kaggle generations predate this script, so no retrospective execution is
claimed. Rebuild quantitative analysis with `python scripts/analyze_results.py`.

Persona v2 is a prepared experiment. `persona_lora_v2.yaml` uses r16/alpha32,
seven attention/MLP targets, LR 2e-4, FP32, effective batch 8, three epochs,
cosine scheduling and 10% warmup. A 1e-4 comparison and v1 reproduction form
the three-run sweep. Prepare it with:

```bash
python scripts/prepare_kaggle_stage.py persona_v2
```

The optional 150-update HH beta configs are
`configs/dpo_hh_pilot_beta005.yaml` and `configs/dpo_hh_pilot_beta03.yaml`.

## Persona v3 follow-up

The historical reward-metric rule selected v2-lr2e-4 before its judgments were
available. After those judgments revealed degeneration, the revised rule in
`docs/persona_selection_rule_v2.md` was timestamped before persona-v3 generation
or judgment. It selects only from greedy generations on the 22 disjoint held-out
validation prompts; final evaluation prompts and judge decisions are excluded.

Prepare and push the combined sequential training/generation job with:

```powershell
.\.venv-local\Scripts\python.exe scripts\prepare_kaggle_stage.py persona_v3
.\.venv-kaggle\Scripts\kaggle.exe kernels push -p kaggle_stages/persona_v3
```

The v3a--d recipes and corresponding evaluation-generation configs are under
`configs/`. TRL 0.18.1 exposes `rpo_alpha`, so v3d uses the specified RPO
regularizer rather than the fallback attention-only adapter. After download,
`scripts/select_persona_adapter.py` writes the deterministic selection record.

## Plotting, report and evidence

```bash
python scripts/plot_metrics.py --metrics /kaggle/working/outputs/sft/metrics.jsonl /kaggle/working/outputs/dpo_hh/metrics.jsonl /kaggle/working/outputs/dpo_ultrafeedback/metrics.jsonl --output /kaggle/working/outputs/plots
```

Plots are derived from measured journals only. Notebooks are thin wrappers.
The SFT, persona, and both full DPO runs have measured artifacts. All four
fixed-set blinded evaluations are complete and have 100% coverage. The packaged
evidence includes prepared-data source revisions, filtering counts, subset
hashes, generation manifests, raw judgments, attempt journals, and recomputed
summaries. See `docs/experiment_log.md` for stage IDs and
`docs/artifact_links.md` for exact saved versions. The report generator rejects
unexpected evaluation counts and builds the <=8-page report only from saved
artifacts.

## References consulted

- Assignment: `docs/assignment_manual.pdf`, pp. 7–23; supplied implementation plan.
- [TRL 0.18.1 DPOTrainer](https://huggingface.co/docs/trl/v0.18.1/en/dpo_trainer),
  plus installed 0.18.1 constructor and disabled-adapter reference context.
- [PEFT LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora),
  plus installed PEFT 0.15.2 API.
- [Kaggle Benchmarks quick start](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md).
- Dataset cards: [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
  [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf),
  [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized),
  [AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval).

The custom main-track math/loops were implemented locally from the assignment;
no completed public SFT/DPO solution was copied. Disclose AI assistance according
to the course's academic-integrity requirements.

### Post-run generation verification (16 October 2026)

The manual requested a 2–5 prompt sanity check before full generation. No
preserved record proves that sequencing occurred for the submitted generations.
A clearly labelled two-prompt post-run verification loaded the exact submitted
persona adapter (SHA-256
`fb832c106783b25d99ba94b184af11c01461df8e0378edcc85652e398fc0444f`),
the pinned SmolLM2 revision `a10cc1512eabd3dde888204e902eca88bddb4951`,
the saved chat template, and greedy decoding with `max_new_tokens=256` and
repetition penalty 1.0. Both responses reached 256 generated tokens without EOS
and appeared truncated; the first was visibly repetitive. Full prompts and raw
responses are saved in `outputs/analysis/postrun_sanity_persona_v1.json`. This
post-run check does not retroactively satisfy the original sequencing rule and
did not replace any submitted generation.

The analysis field `retokenized_decoded_length_ge_256_pct` encodes saved decoded
text with the pinned tokenizer and reports the fraction with length at least
256. It is a proxy for reaching the generation limit because special-token
removal, decoding, and re-tokenization can change token counts; it is not called
a confirmed cap-hit or truncation rate.

## Final v2/report commands

```powershell
$env:PYTHONPATH='.'
.venv-local\Scripts\python.exe scripts/check_empty_prompts.py
.venv-local\Scripts\python.exe scripts/compare_persona_runs.py
.venv-local\Scripts\python.exe scripts/analyze_results.py
.venv-local\Scripts\python.exe scripts/build_report_tex.py
.venv-local\Scripts\python.exe scripts/check_report_numbers.py
Push-Location report; latexmk -pdf -interaction=nonstopmode main.tex; Pop-Location
.venv-local\Scripts\python.exe -m pytest -q
.venv-local\Scripts\python.exe scripts/assemble_evidence_v2.py
.venv-local\Scripts\python.exe scripts/build_submission_zip.py
.venv-local\Scripts\python.exe scripts/validate_submission.py
```

Kaggle persona generation v2 runs the saved three-prompt sanity check before full generation. Judging uses blinded mappings, the same v1 rubrics and pair builder, `google/gemini-3.6-flash`, batch size five, a five-second minimum delay, at most three attempts, and resumable journals.

### Persona-v3 completed results

`moibbi/pa2-persona-v3-full` v1 completed. The revised held-out rule selected v3b before judging. Kaggle Benchmark task `pa2-judge-persona-v3` v1 completed as run 4809623. The displayed runner was Gemini 3.8 Flash; every judgment manifest records the fixed actual judge `google/gemini-3.6-flash`. All eight tracks have 20/20 coverage, zero unknowns/failures, and saved attempts at least five seconds apart. The machine-readable 10,000-resample, seed-42 intervals are in `outputs/analysis/persona_v3_judging_summary.json`.

### HH beta generation check

`moibbi/pa2-hh-beta-generation` v1 completed the three matched 150-update
pilots, pre-generation sanity checks, and 300 fixed-prompt generations per
beta. The two resumable judge tasks completed as runs 4888390 and 4888391.
Recompute the saved comparison summary with:

```powershell
$env:PYTHONPATH='.'
.venv-local\Scripts\python.exe scripts/analyze_hh_beta_generation.py
.venv-local\Scripts\python.exe scripts/assemble_evidence_beta.py
```

The generation, sanity, training, pair, and judgment artifacts are
hash-manifested under `submission/evidence_beta/`.
