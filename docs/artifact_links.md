# Saved Kaggle artifacts and local evidence

These are actual private Kaggle jobs; privacy settings may prevent another
account from opening the links. Local downloaded metadata and small evidence are
packaged separately for review. A kernel is counted as evidence only after its
completion status and saved output have been checked.

The version labels in the table identify the exact successful saved versions.
The links may resolve to the latest private view only, and grader access has not
been verified from another account. Full-model checkpoints remain at the output
paths shown below; their SHA-256 metadata is in the packaged stage manifests.

| Stage | Kaggle artifact | Current evidence |
|---|---|---|
| Integrated T4x2 pilot v3 | [pilot](https://www.kaggle.com/code/moibbi/pa2-small-gpu-pilot) | `outputs/pilot-v3-artifacts/pa2-pilot/pilot_result.json`; COMPLETE, `ok: true` |
| Full SFT v3 | [SFT](https://www.kaggle.com/code/moibbi/pa2-sft-full) | `outputs/sft-full-v3/outputs/sft/`; COMPLETE, 938 updates |
| Persona LoRA-DPO v2 | [persona](https://www.kaggle.com/code/moibbi/pa2-persona-full) | `outputs/persona-full-v2/outputs/persona/`; COMPLETE, adapter reload verified |
| Persona generation v1 | [persona generations](https://www.kaggle.com/code/moibbi/pa2-generate-persona) | `outputs/persona-generation-v1/outputs/generations/`; COMPLETE, 20 aligned prompts |
| HH DPO v1 | [HH](https://www.kaggle.com/code/moibbi/pa2-dpo-hh-full) | `outputs/dpo-hh-v1/outputs/dpo_hh/`; COMPLETE, 469 updates, validation reward accuracy 0.592 |
| UltraFeedback DPO v1 | [UltraFeedback](https://www.kaggle.com/code/moibbi/pa2-dpo-ultrafeedback-full) | `outputs/dpo-uf-v1/outputs/dpo_ultrafeedback/`; COMPLETE, 469 updates, validation reward accuracy 0.701 |
| SFT fixed-set generation v1 | [SFT generation](https://www.kaggle.com/code/moibbi/pa2-generate-sft) | `outputs/generation-sft-v1/outputs/generations/`; COMPLETE, three 300-row sets |
| HH fixed-set generation v1 | [HH generation](https://www.kaggle.com/code/moibbi/pa2-generate-hh) | `outputs/generation-hh-v1/outputs/generations/`; COMPLETE, 300 rows |
| UltraFeedback generation v1 | [UltraFeedback generation](https://www.kaggle.com/code/moibbi/pa2-generate-ultrafeedback) | `outputs/generation-uf-v1/outputs/generations/`; COMPLETE, 300 rows |
| HH judge v1 | [HH task](https://www.kaggle.com/benchmarks/tasks/moibbi/pa2-judge-hh/1) | `outputs/judge-hh-v1/`; COMPLETE, 300/300; 126W/90L/84T |
| UltraFeedback judge v1 | [UltraFeedback task](https://www.kaggle.com/benchmarks/tasks/moibbi/pa2-judge-ultrafeedback/1) | `outputs/judge-ultrafeedback-v1/`; COMPLETE, 300/300; 86W/132L/82T |
| Persona style judge v2 | [style task](https://www.kaggle.com/benchmarks/tasks/moibbi/pa2-judge-persona-style/2) | `outputs/judge-persona-style-v2/`; COMPLETE, 20/20 |
| Persona quality judge v1 | [quality task](https://www.kaggle.com/benchmarks/tasks/moibbi/pa2-judge-persona-quality/1) | `outputs/judge-persona-quality-v1/`; COMPLETE, 20/20 |

Checkpoint locations: SFT v3 `outputs/sft-full-v3/outputs/sft/best/`; HH DPO v1
`outputs/dpo-hh-v1/outputs/dpo_hh/best/`; UltraFeedback DPO v1
`outputs/dpo-uf-v1/outputs/dpo_ultrafeedback/best/`; persona v2
`outputs/persona-full-v2/outputs/persona/adapter/`.

The actual judge calls use `google/gemini-3.6-flash` inside the benchmark task;
the task runner shown by Kaggle is Gemini 3.7 Flash. Every saved decision row
records the actual judge. Full-model weights remain in the saved Kaggle kernel
outputs; the 3.3 MB persona adapter and small evidence can be included in the
repository submission package.

- GPU smoke: moibbi/pa2-cli-gpu-smoke-oct12, version 5, COMPLETE, downloaded outputs/kaggle_preflight/gpu_smoke.

- Persona v2 sweep: moibbi/pa2-persona-v2-full, version 1, COMPLETE (canonical slug assigned by Kaggle).


- Persona v2 generation v1: moibbi/pa2-generate-persona-v2, ERROR; sanity BatchEncoding kwargs fixed in v2.

- HH beta pilots: `moibbi/pa2-dpo-hh-pilots-full` v1 ERROR (checkpoint disk exhaustion); v2 COMPLETE after the metrics-only fix.
- T4x2 environment evidence: moibbi/pa2-env-t4x2-check v2 COMPLETE; v1 failed packaging, v3 notebook execution pending.

- T4x2 environment v1 ERROR (missing packaged requirements); fixed by embedding inputs in v2.


## Persona v2 and environment artifacts

| Artifact | Version/status | Downloaded evidence |
|---|---|---|
| `moibbi/pa2-persona-v2-full` | v1 COMPLETE | `outputs/persona_v2/` |
| `moibbi/pa2-generate-persona-v2` | v1 ERROR, v2 COMPLETE | `outputs/persona_v2_generation/` |
| `moibbi/pa2-judge-persona-v2-style` | v2 COMPLETE, run 4802960 | `outputs/persona_v2_evaluation/persona_lora_v2/` |
| `moibbi/pa2-judge-persona-v2-quality` | v1 COMPLETE, run 4802966 | same |
| `moibbi/pa2-judge-persona-v2-lr1e4-style` | v1 COMPLETE, run 4802970 | `outputs/persona_v2_evaluation/persona_lora_v2_lr1e4/` |
| `moibbi/pa2-judge-persona-v2-lr1e4-quality` | v1 COMPLETE, run 4802972 | same |
| `moibbi/pa2-env-t4x2-check` | v1 ERROR; v2 and v3 COMPLETE | `outputs/analysis/env_kaggle/` |
| `moibbi/pa2-dpo-hh-pilots-full` | v2 COMPLETE | `outputs/hh_beta_pilots_v2/` and `outputs/analysis/hh_beta_pilots.json` |
| `moibbi/pa2-hh-beta-generation` | v1 COMPLETE | `outputs/hh_beta_generation/`; three 300-row generation sets |
| `moibbi/pa2-judge-hh-beta005-vs-beta01` | v1 COMPLETE, run 4888390 | `outputs/hh_beta_judging/`; 300/300, 67W/56L/177T |
| `moibbi/pa2-judge-hh-beta03-vs-beta01` | v1 COMPLETE, run 4888391 | `outputs/hh_beta_judging/`; 300/300, 77W/71L/152T |

The persona benchmark runner label was Gemini 3.8 Flash; the judge invoked by the task code and recorded in every manifest/row was `google/gemini-3.6-flash`. Environment v2 recorded two Tesla T4 GPUs, torch 2.11.0+cu128, Transformers 4.51.3, Datasets 3.6.0, TRL 0.18.1, PEFT 0.15.2, and Accelerate 1.6.0.

- HH beta pilots v1 ERROR on checkpoint disk use; the metrics-only v2 retry is COMPLETE and its saved validation metrics are included locally.

- HH beta pilots: moibbi/pa2-dpo-hh-pilots-full v1 ERROR (checkpoint disk exhaustion), v2 COMPLETE; metrics in outputs/hh_beta_pilots_v2 and analysis in outputs/analysis/hh_beta_pilots.json.

| `moibbi/pa2-persona-v3-full` | v1 pushed 2026-10-16T18:28:40.8718859+05:00; status not polled | expected local download: `outputs/persona_v3/` |

| Kaggle Benchmark task `pa2-judge-persona-v3` | initial version pushed 2026-10-16T21:55:10.6136943+05:00; server state VALIDATING at creation | expected download: `outputs/persona_v3_judging/` |

| Kaggle Benchmark `pa2-judge-persona-v3` | v1 COMPLETE; run 4809623 | `outputs/persona_v3_judging/`, `submission/evidence_v3/judging/` |

| `moibbi/pa2-executed-environment-notebook` | v1 pushed 2026-10-17T16:02:06.6422661+05:00 | expected output: `00_setup_and_env_check.executed.ipynb` |

| `moibbi/pa2-executed-environment-notebook` | v2 pushed 2026-10-17T16:24:25.552591+05:00; fixes extracted-dataset layout | expected output: `00_setup_and_env_check.executed.ipynb` |

| `moibbi/pa2-executed-environment-notebook` | v3 pushed 2026-10-17T16:53:42.226676+05:00; corrected runner restarted after manual v2 stop | expected output: `00_setup_and_env_check.executed.ipynb` |

| `moibbi/pa2-executed-environment-notebook` | v4 pushed 2026-10-17T18:41:03.905382+05:00; fresh rerun explicitly requested after prolonged v3 queue | expected output: `00_setup_and_env_check.executed.ipynb` |

| `moibbi/pa2-executed-environment-notebook` | v5 pushed 2026-10-17T18:44:23.413393+05:00; README-safe install after v4 dependency failure | expected output: `00_setup_and_env_check.executed.ipynb` |

| `moibbi/pa2-executed-environment-notebook` | v5 COMPLETE | authentic executed `notebooks/00_setup_and_env_check.ipynb`; downloaded source under `outputs/analysis/executed_env_v5/` |
