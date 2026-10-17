"""Generate report macros, tables, and figures only from saved evidence."""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tex(value):
    return str(value).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def main():
    analysis = read("outputs/analysis/analysis.json")
    persona = read("outputs/analysis/persona_comparison.json")
    env = read("outputs/analysis/env_kaggle/env_kaggle_t4x2.json")
    report = Path("report"); (report / "figures").mkdir(parents=True, exist_ok=True)
    ev, cis = analysis["evaluations"], analysis["bootstrap_95pct_ci"]
    unique = analysis["hh"]["unique_prompts"]
    revised_persona = next(row for row in persona["runs"] if row["run"] == persona["revised_selected"])
    values = {
        "HHW": ev["hh"]["win"], "HHL": ev["hh"]["loss"], "HHT": ev["hh"]["tie"],
        "HHScore": f'{100*ev["hh"]["tie_adjusted"]:.1f}',
        "UFW": ev["ultrafeedback"]["win"], "UFL": ev["ultrafeedback"]["loss"], "UFT": ev["ultrafeedback"]["tie"],
        "UFScore": f'{100*ev["ultrafeedback"]["tie_adjusted"]:.1f}',
        "HHUniqueW": unique["win"], "HHUniqueL": unique["loss"], "HHUniqueT": unique["tie"],
        "HHUniqueN": unique["n"], "HHUniqueScore": f'{100*unique["tie_adjusted"]:.1f}',
        "HHUniqueCILo": f'{100*unique["bootstrap_95pct_ci"][0]:.1f}',
        "HHUniqueCIHi": f'{100*unique["bootstrap_95pct_ci"][1]:.1f}',
        "HHRho": f'{analysis["hh"]["spearman_delta_vs_outcome"]:.3f}',
        "HHNetWins": ev["hh"]["win"]-ev["hh"]["loss"],
        "UFNetLosses": ev["ultrafeedback"]["loss"]-ev["ultrafeedback"]["win"],
        "GPUCount": env["cuda_device_count"], "GPUName": env["cuda_device_names"][0],
        "TorchVersion": env["versions"]["torch"], "TransformersVersion": env["versions"]["transformers"],
        "TRLVersion": env["versions"]["trl"], "PEFTVersion": env["versions"]["peft"],
        "PersonaSelected": revised_persona["run"],
        "PersonaStyleW": revised_persona["style"]["W"], "PersonaStyleL": revised_persona["style"]["L"],
        "PersonaStyleT": revised_persona["style"]["T"],
        "PersonaStyleScore": f'{100*revised_persona["style"]["tie_adjusted"]:.1f}',
        "PersonaStyleCILo": f'{100*revised_persona["style"]["bootstrap_95pct_ci"][0]:.1f}',
        "PersonaStyleCIHi": f'{100*revised_persona["style"]["bootstrap_95pct_ci"][1]:.1f}',
        "PersonaQualityW": revised_persona["quality"]["W"], "PersonaQualityL": revised_persona["quality"]["L"],
        "PersonaQualityT": revised_persona["quality"]["T"],
        "PersonaQualityScore": f'{100*revised_persona["quality"]["tie_adjusted"]:.1f}',
        "PersonaQualityCILo": f'{100*revised_persona["quality"]["bootstrap_95pct_ci"][0]:.1f}',
        "PersonaQualityCIHi": f'{100*revised_persona["quality"]["bootstrap_95pct_ci"][1]:.1f}',
    }
    macro_lines = []
    for key, value in values.items():
        suffix = r"\xspace" if key == "PersonaSelected" else ""
        macro_lines.append(f"\\newcommand{{\\{key}}}{{{tex(value)}{suffix}}}\n")
    (report / "generated_numbers.tex").write_text("".join(macro_lines), encoding="utf-8")
    historical = revised_persona
    headline_rows = []
    for label, row, ci, candidate, baseline in (
        ("HH required rows", ev["hh"], cis["hh"], "DPO", "SFT"),
        ("HH unique prompts", unique, unique["bootstrap_95pct_ci"], "DPO", "SFT"),
        ("UltraFeedback/AlpacaEval", ev["ultrafeedback"], cis["ultrafeedback"], "DPO", "SFT"),
        (f"Persona {historical['run']} style", historical["style"], historical["style"]["bootstrap_95pct_ci"], "adapter", "base"),
        (f"Persona {historical['run']} quality", historical["quality"], historical["quality"]["bootstrap_95pct_ci"], "adapter", "base")):
        score = row["tie_adjusted"]
        win, loss, tie = (row.get("win", row.get("W")), row.get("loss", row.get("L")),
                          row.get("tie", row.get("T")))
        better = candidate if score > .5 else baseline if score < .5 else "tie"
        headline_rows.append(f"{label} & {win}/{loss}/{tie} & "
                             f"{100*score:.1f}\\% [{100*ci[0]:.1f},{100*ci[1]:.1f}] & {better}\\\\")
    (report / "generated_headline.tex").write_text(
        "\\begin{table}[H]\\centering\\footnotesize\\caption{Headline blinded results; sets are reported separately.}\\label{tab:headline}\n"
        "\\begin{tabularx}{\\textwidth}{Xrrl}\\toprule Evaluation & W/L/T & Tie-adjusted score (95\\% CI) & Better model\\\\\\midrule\n"
        + "\n".join(headline_rows) + "\n\\bottomrule\\end{tabularx}\\end{table}\n", encoding="utf-8")
    tables = []
    sft = read("submission/evidence/stages/sft/resolved_config.json")
    hh = read("submission/evidence/stages/dpo_hh/resolved_config.json")
    uf = read("submission/evidence/stages/dpo_ultrafeedback/resolved_config.json")
    persona_manifest = read("submission/evidence/stages/persona/manifest.json")
    pv1 = persona_manifest["config"]
    pv2 = read("configs/persona_lora_v2.yaml")
    generation = read("configs/generate_persona_lora_v2.yaml")
    prompt_audit = read("outputs/analysis/empty_prompt_audit.json")
    audit = {(r["track"], r["split"]): r for r in prompt_audit["splits"]}
    ref_hash = hh["initial_checkpoint"]["model.safetensors"][:12]
    tables += ["\\begin{table*}[t]\\centering\\scriptsize\\caption{Resolved experimental configuration. Eff. batch is microbatch $\\times$ accumulation.}",
               "\\setlength{\\tabcolsep}{2pt}\\begin{tabularx}{\\textwidth}{lXXXXXX}\\toprule Track & Seed/model & LoRA & Optim./sched. & LR; warmup; decay; clip & Batch; precision; length & Epoch/update; $\\beta$; reference\\\\\\midrule",
               f"SFT & {sft['seed']}; Qwen2.5-0.5B & none & AdamW; {sft['scheduler']} & {sft['learning_rate']:.0e}; {sft['warmup_ratio']}; {sft['weight_decay']}; {sft['clip_norm']} & {sft['microbatch']}x{sft['accumulation']}={sft['microbatch']*sft['accumulation']}; fp16; {sft['max_length']} & {sft['epochs']} epoch; --; base rev. {sft['revision'][:12]}\\\\",
               f"HH DPO & {hh['seed']}; SFT policy & none & AdamW; constant & {hh['learning_rate']:.0e}; {hh['warmup_ratio']}; {hh['weight_decay']}; {hh['clip_norm']} & {hh['microbatch']}x{hh['accumulation']}={hh['microbatch']*hh['accumulation']}; fp16; {hh['max_prompt_length']}/{hh['max_length']} & {hh['epochs']} epoch; {hh['beta']}; SHA {ref_hash}\\\\",
               f"UF DPO & {uf['seed']}; SFT policy & none & AdamW; constant & {uf['learning_rate']:.0e}; {uf['warmup_ratio']}; {uf['weight_decay']}; {uf['clip_norm']} & {uf['microbatch']}x{uf['accumulation']}={uf['microbatch']*uf['accumulation']}; fp16; {uf['max_prompt_length']}/{uf['max_length']} & {uf['epochs']} epoch; {uf['beta']}; SHA {ref_hash}\\\\",
               f"Persona v1 & {pv1['seed']}; SmolLM2 & r={pv1['rank']}, a={pv1['alpha']}; q/v & AdamW; constant & {pv1['learning_rate']:.0e}; 0; 0; -- & {pv1['microbatch']}x{pv1['accumulation']}={pv1['microbatch']*pv1['accumulation']}; fp32; {pv1['max_length']} & {pv1['epochs']} epoch/12; {pv1['beta']}; base adapters off\\\\",
               f"Persona v2 & {pv2['seed']}; SmolLM2 & r={pv2['rank']}, a={pv2['alpha']}; attention+MLP & AdamW; {pv2['scheduler']} & {pv2['learning_rate']:.0e}; {pv2['warmup_ratio']}; {pv2['weight_decay']}; -- & {pv2['microbatch']}x{pv2['accumulation']}={pv2['microbatch']*pv2['accumulation']}; {pv2['precision']}; {pv2['max_length']} & {pv2['epochs']} epochs/72; {pv2['beta']}; base adapters off\\\\",
               "\\bottomrule\\end{tabularx}\\end{table*}",
               "\\begin{table*}[t]\\centering\\scriptsize\\caption{Guide alignment and evidence-backed deviations.}",
               "\\setlength{\\tabcolsep}{2pt}\\begin{tabularx}{\\textwidth}{p{1.6cm}p{1.3cm}p{1.8cm}XXXp{1.5cm}}\\toprule Parameter & Guide & Final & Reason & Evidence & Observed effect & Cost\\\\\\midrule",
               "SFT/main DPO & prescribed & matched & preserve assignment comparison & resolved configs and metrics & attribution remains within recipe & not ablated\\\\",
               "Persona model & SmolLM2-360M & matched & manual requirement & pinned model revision & valid lightweight PEFT track & 360M base\\\\",
               "LoRA shape & v1 r8/a16 q,v & v2 r16/a32 attention+MLP & v1 had little behavioral movement & v1: 12 updates and 15/20 identical & v2 changed 20/20 but degenerated & 0.82M to 8.68M trainable\\\\",
               "Persona LR/epochs & v1 5e-5/1 & v2 2e-4 and 1e-4/3 & test stronger exposure & held-out metrics and blinded judgments & margins rose; generation quality fell & 72 updates each\\\\",
               "Precision & fp16 pilot & fp32 & avoid nonfinite gradients & preserved pilot log & finite weights and reload passed & slower, more stable\\\\",
               "Batch/eval & effective 16; epoch eval & effective 8; epoch eval & fit wider adapters and diagnose each epoch & resolved configs/metric logs & not ablated & lower effective batch\\\\",
               "DPO optimizer & open & AdamW & standard implementation choice & source and resolved configs & not ablated & same main-track optimizer\\\\",
               "HH beta & 0.1 & 0.1 submitted; pilots 0.05/0.1/0.3 & validation plus 300-prompt generation check & pilot metrics and blinded judgments & 150 updates per pilot\\\\",
               "SFT final window & batch 32 target & 16 examples & consume all examples once & metric journal & complete 30,000-row epoch & no padding duplicate\\\\",
               "\\bottomrule\\end{tabularx}\\end{table*}",
               "\\begin{table}[t]\\centering\\footnotesize\\caption{Frozen prepared data audit and generation settings.}",
               "\\begin{tabular}{lrrr}\\toprule Track & Train & Validation & Empty formatted\\\\\\midrule",
               f"SFT & {audit[('sft','train')]['examples']:,} & {audit[('sft','validation')]['examples']:,} & {audit[('sft','train')]['empty_formatted_strings'] + audit[('sft','validation')]['empty_formatted_strings']}\\\\",
               f"HH & {audit[('hh','train')]['examples']:,} & {audit[('hh','validation')]['examples']:,} & {audit[('hh','train')]['empty_formatted_strings'] + audit[('hh','validation')]['empty_formatted_strings']}\\\\",
               f"UltraFeedback & {audit[('ultrafeedback','train')]['examples']:,} & {audit[('ultrafeedback','validation')]['examples']:,} & {audit[('ultrafeedback','train')]['empty_formatted_strings'] + audit[('ultrafeedback','validation')]['empty_formatted_strings']}\\\\",
               f"Persona & {persona_manifest['counts']['valid']-pv2['validation_size']} & {pv2['validation_size']} & 0\\\\",
               f"\\bottomrule\\end{{tabular}}\\par\\vspace{{2pt}}\\scriptsize Greedy decoding; batch {generation['batch_size']}; max {generation['max_new_tokens']} new tokens; repetition penalty {generation['repetition_penalty']}; pinned chat templates.\\end{{table}}"]
    tables += ["\\begin{table*}[t]\\centering\\small\\caption{Blinded evaluation; score is tie-adjusted with row-bootstrap 95\\% CI.}\\label{tab:blinded}",
               "\\begin{tabular}{lrrrrllllr}\\toprule Track & N & W & L & T & Score & Decisive & Raw win & 95\\% CI & Retry/U/F\\\\\\midrule"]
    summaries = {
        "hh": read("submission/evidence/evaluation/hh_summary.json")["hh"],
        "ultrafeedback": read("submission/evidence/evaluation/ultrafeedback_summary.json")["ultrafeedback"],
        "persona_style": read("submission/evidence/evaluation/persona_style_summary.json")["persona_style"],
        "persona_quality": read("submission/evidence/evaluation/persona_quality_summary.json")["persona_quality"],
    }
    for key, label in (("hh", "HH rows"), ("ultrafeedback", "UltraFeedback"),
                       ("persona_style", "Persona v1 style"), ("persona_quality", "Persona v1 quality")):
        row, ci = ev[key], cis[key]
        summary = summaries[key]
        tables.append(f"{label} & {row['n']} & {row.get('win',0)} & {row.get('loss',0)} & {row.get('tie',0)} & {100*row['tie_adjusted']:.1f}\\% & {100*summary['decisive_win_rate']:.1f}\\% & {100*summary['raw_win_rate']:.1f}\\% & [{100*ci[0]:.1f},{100*ci[1]:.1f}]\\% & {summary['pair_retries_including_failures']}/{summary['U']}/{summary['F']}\\\\")
    tables += [f"HH unique prompts & {unique['n']} & {unique['win']} & {unique['loss']} & {unique['tie']} & {100*unique['tie_adjusted']:.1f}\\% & {100*unique['win']/(unique['win']+unique['loss']):.1f}\\% & {100*unique['win']/unique['n']:.1f}\\% & [{100*unique['bootstrap_95pct_ci'][0]:.1f},{100*unique['bootstrap_95pct_ci'][1]:.1f}]\\% & --\\\\",
               "\\bottomrule\\end{tabular}\\end{table*}"]
    tables += ["\\begin{table*}[t]\\centering\\scriptsize\\caption{All seven persona adapters. $^\\dagger$ is the historical-rule selection; bold is the revised-rule selection. Held-out generation uses 22 disjoint validation prompts; judge CIs use 10,000 seed-42 bootstrap resamples.}\\label{tab:persona-all}",
               "\\textbf{(a) Training and held-out preference metrics}\\par\\vspace{2pt}",
               "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrrr}\\toprule Variant & Params & Updates & Epoch & LR & $\\beta$ & RPO & Accuracy & Margin & Eval loss\\\\\\midrule"]
    for row in persona["runs"]:
        label = f"\\textbf{{{tex(row['run'])}}}" if row['run'] == persona.get('revised_selected') else (f"{tex(row['run'])}$^\\dagger$" if row['run'] == persona.get('historical_selected') else tex(row['run']))
        rpo = "--" if row.get('rpo_alpha') is None else f"{row['rpo_alpha']:.1f}"
        tables.append(f"{label} & {row['trainable_parameters']:,} & {row['updates']} & {row['epochs_recorded']:.3f} & {row['learning_rate']:.0e} & {row['beta']:.1f} & {rpo} & {row['held_out_reward_accuracy']:.3f} & {row['held_out_reward_margin']:.3f} & {row['eval_loss']:.3f}\\\\")
    tables += ["\\bottomrule\\end{tabular}}\\par\\vspace{5pt}\\textbf{(b) Held-out selection behavior, final evaluation behavior, and blinded judgments}\\par\\vspace{2pt}",
               "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrr}\\toprule Variant & Held deg./chg./marker & Held words & Eval chg. & Eval deg./ident. & Eval words & Style W/L/T; score (CI) & Quality W/L/T; score (CI)\\\\\\midrule"]
    for row in persona["runs"]:
        s, q = row["style"], row["quality"]
        style = (f"{s['W']}/{s['L']}/{s['T']}; {100*s['tie_adjusted']:.1f} [{100*s['bootstrap_95pct_ci'][0]:.1f},{100*s['bootstrap_95pct_ci'][1]:.1f}]" if s else "pending")
        quality = (f"{q['W']}/{q['L']}/{q['T']}; {100*q['tie_adjusted']:.1f} [{100*q['bootstrap_95pct_ci'][0]:.1f},{100*q['bootstrap_95pct_ci'][1]:.1f}]" if q else "pending")
        hd=100*row.get('held_out_degeneration_rate',0); hc=100*row.get('held_out_changed_vs_base_rate',0); hm=100*row.get('held_out_style_marker_rate',0); hw=row.get('held_out_mean_words',0)
        label = f"\\textbf{{{tex(row['run'])}}}" if row['run'] == persona.get('revised_selected') else (f"{tex(row['run'])}$^\\dagger$" if row['run'] == persona.get('historical_selected') else tex(row['run']))
        tables.append(f"{label} & {hd:.1f}/{hc:.1f}/{hm:.1f} & {hw:.1f} & {row['outputs_changed_vs_base']}/{row['n']} & {row['degeneration_pct']:.1f}/{row['identical_to_base_pct']:.1f} & {row['mean_words']:.1f} & {style} & {quality}\\\\")
    tables += ["\\bottomrule\\end{tabular}}\\end{table*}",
               "\\begin{table}[t]\\centering\\footnotesize\\caption{HH outcomes by DPO-minus-SFT length tercile.}\\label{tab:hh-length}\\begin{tabular}{lrrr}\\toprule Tercile & W & L & T\\\\\\midrule"]
    for name, row in analysis["hh"]["token_length_delta_terciles"].items():
        tables.append(f"{name} ({row['delta_min']}..{row['delta_max']}) & {row.get('win',0)} & {row.get('loss',0)} & {row.get('tie',0)}\\\\")
    tables += ["\\bottomrule\\end{tabular}\\end{table}",
               "\\begin{table}[t]\\centering\\footnotesize\\caption{HH outcomes by repetition state.}\\label{tab:hh-repetition}\\begin{tabular}{lrrrr}\\toprule State & N & W & L & T\\\\\\midrule"]
    for name, row in analysis["hh"]["outcome_by_degeneration"].items():
        tables.append(f"{name} & {row['n']} & {row.get('win',0)} & {row.get('loss',0)} & {row.get('tie',0)}\\\\")
    tables += ["\\bottomrule\\end{tabular}\\end{table}"]
    examples = read("outputs/analysis/report_examples.json")
    tables += ["\\begin{table*}[t]\\centering\\footnotesize\\caption{Deterministically selected judged examples. Text is exact after whitespace normalization; an ellipsis marks sentence-boundary truncation.}\\label{tab:examples}",
               "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{2.1cm}>{\\raggedright\\arraybackslash}p{1.35cm}X}\\toprule Track & Outcome & Saved text and analysis\\\\\\midrule"]
    for track in ("hh", "ultrafeedback", "persona_v3b"):
        if track not in examples:
            continue
        for outcome in ("win", "loss", "tie"):
            row = examples[track][outcome]
            detail = (f"\\textbf{{Prompt}} ({tex(row['prompt_id'])}): {tex(row['prompt'])} "
                      f"\\textbf{{{tex(row.get('candidate_label','DPO'))}}}: {tex(row['candidate'])} "
                      f"\\textbf{{{tex(row.get('baseline_label','SFT'))}}}: {tex(row['baseline'])} "
                      f"\\emph{{{tex(row['analysis'])}}}")
            tables.append(f"{tex(track)} & {outcome} & {detail}\\\\")
    tables += ["\\bottomrule\\end{tabularx}\\end{table*}"]
    beta_path = Path("outputs/analysis/hh_beta_pilots.json")
    if beta_path.exists():
        tables += ["\\begin{table}[t]\\centering\\footnotesize\\caption{Matched HH beta pilots.}\\label{tab:hh-beta}\\begin{tabular}{rrrrr}\\toprule $\\beta$ & Update & Accuracy & Margin & Loss\\\\\\midrule"]
        for run in read(beta_path)["runs"]:
            for point in run["validation"]:
                tables.append(f"{run['beta']:.2f} & {point['step']} & {point['reward_accuracy']:.3f} & {point['reward_margin']:.3f} & {point['loss']:.3f}\\\\")
        tables += ["\\bottomrule\\end{tabular}\\end{table}"]
    combined = "\n".join(tables) + "\n"
    (report / "generated_tables.tex").write_text(combined, encoding="utf-8")
    import re
    pieces = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", combined, flags=re.S)
    groups = {"configuration": [], "results": [], "persona": [], "diagnostics": [], "examples": []}
    for piece in pieces:
        piece = piece.replace("\\begin{table*}", "\\begin{table}").replace(
            "\\end{table*}", "\\end{table}").replace("[t]", "[H]", 1)
        if "Resolved experimental configuration" in piece or "Frozen prepared data" in piece:
            groups["configuration"].append(piece)
        elif "Guide alignment" in piece or "beta pilots" in piece:
            groups["diagnostics"].append(piece)
        elif "Blinded evaluation" in piece:
            groups["results"].append(piece)
        elif "persona variants" in piece or "persona adapters" in piece or "Persona generation" in piece:
            groups["persona"].append(piece)
        elif "Deterministically selected judged examples" in piece:
            continue
        else:
            groups["examples"].append(piece)
    example_prose = ["\\subsection*{Representative judged examples}"]
    for track in ("hh", "ultrafeedback", "persona_v3b"):
        if track not in examples:
            continue
        for outcome in ("win", "loss", "tie"):
            row = examples[track][outcome]
            example_prose.append(
                f"\\paragraph{{{tex(track)} {outcome} ({tex(row['prompt_id'])}).}} "
                f"\\textbf{{Prompt:}} {tex(row['prompt'])} "
                f"\\textbf{{{tex(row.get('candidate_label','DPO'))}:}} {tex(row['candidate'])} "
                f"\\textbf{{{tex(row.get('baseline_label','SFT'))}:}} {tex(row['baseline'])} "
                f"\\emph{{{tex(row['analysis'])}}}")
    groups["examples"].extend(example_prose)
    for name, items in groups.items():
        (report / f"generated_{name}.tex").write_text("\n".join(items) + "\n", encoding="utf-8")
    from report_revision import revise
    revise()
    print(json.dumps({"macros": len(values), "persona_runs": len(persona["runs"]), "beta_included": beta_path.exists()}))


if __name__ == "__main__":
    main()
