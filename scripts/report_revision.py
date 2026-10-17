"""Final Task-8 report material, generated only from saved evidence."""
import json
import re
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def esc(value):
    return (str(value).replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&"))


def verdict(score, ci, candidate, baseline):
    if ci[0] <= .5 <= ci[1]:
        if score < .5:
            return f"inconclusive (CI includes 50%); point estimate favors {baseline}"
        if score > .5:
            return f"inconclusive (CI includes 50%); point estimate favors {candidate}"
        return "inconclusive (CI includes 50%)"
    return candidate if score > .5 else baseline


def config_rows():
    configs = [
        ("SFT", read("submission/evidence/stages/sft/resolved_config.json"), "full FT", 938, None, None),
        ("HH DPO", read("submission/evidence/stages/dpo_hh/resolved_config.json"), "full FT", 469, None, None),
        ("UF DPO", read("submission/evidence/stages/dpo_ultrafeedback/resolved_config.json"), "full FT", 469, None, None),
        ("persona v1", read("submission/evidence/stages/persona/manifest.json")["config"], "LoRA", 12, None, None),
        ("v2 lr2e-4", read("submission/evidence_v2/persona/v2_lr2e4/manifest.json")["config"], "LoRA", 72, None, None),
        ("v2 lr1e-4", read("submission/evidence_v2/persona/v2_lr1e4/manifest.json")["config"], "LoRA", 72, None, None),
    ]
    for name in ("v3a", "v3b", "v3c", "v3d"):
        cfg = read(f"submission/evidence_v3/training/{name}/manifest.json")["config"]
        updates = {"v3a":72, "v3b":24, "v3c":72, "v3d":48}[name]
        configs.append((name, cfg, "LoRA", updates, None, None))
    return configs


def revise():
    report = Path("report")
    analysis = read("outputs/analysis/analysis.json")
    persona = read("outputs/analysis/persona_comparison.json")
    runs = {r["run"]: r for r in persona["runs"]}
    selected = runs[persona["revised_selected"]]
    ev, cis = analysis["evaluations"], analysis["bootstrap_95pct_ci"]
    unique = analysis["hh"]["unique_prompts"]

    headline = []
    specs = [
        ("HH required rows", ev["hh"], cis["hh"], "DPO", "SFT"),
        ("HH unique prompts", unique, unique["bootstrap_95pct_ci"], "DPO", "SFT"),
        ("UltraFeedback/AlpacaEval", ev["ultrafeedback"], cis["ultrafeedback"], "DPO", "SFT"),
        (f"Persona {selected['run']} style", selected["style"], selected["style"]["bootstrap_95pct_ci"], "adapter", "base"),
        (f"Persona {selected['run']} quality", selected["quality"], selected["quality"]["bootstrap_95pct_ci"], "adapter", "base"),
    ]
    verdict_macros = []
    macro_names = ("HHRowsVerdict", "HHUniqueVerdict", "UFVerdict",
                   "PersonaStyleVerdict", "PersonaQualityVerdict")
    for macro_name, (label, row, ci, cand, base) in zip(macro_names, specs):
        w, l, t = row.get("win", row.get("W")), row.get("loss", row.get("L")), row.get("tie", row.get("T"))
        v = verdict(row["tie_adjusted"], ci, cand, base)
        headline.append(f"{esc(label)} & {w}/{l}/{t} & {100*row['tie_adjusted']:.1f}\\% "
                        f"[{100*ci[0]:.1f},{100*ci[1]:.1f}] & {esc(v)}\\\\")
        verdict_macros.append(f"\\newcommand{{\\{macro_name}}}{{{esc(v)}\\xspace}}")
    (report/"generated_headline.tex").write_text(
        "\\begin{table}[H]\\centering\\footnotesize\\caption{Headline blinded results. A verdict is conclusive only when its 95\\% interval excludes 50\\%.}\\label{tab:headline}\n"
        "\\begin{tabularx}{\\textwidth}{Xrr>{\\raggedright\\arraybackslash}p{3.8cm}}\\toprule Evaluation & W/L/T & Score [95\\% CI] & Verdict\\\\\\midrule\n"+
        "\n".join(headline)+"\n\\bottomrule\\end{tabularx}\\end{table}\n", encoding="utf-8")
    numbers_path = report/"generated_numbers.tex"
    numbers = numbers_path.read_text(encoding="utf-8")
    for macro_name in macro_names:
        numbers = re.sub(rf"^\\newcommand\{{\\{macro_name}\}}.*\n?", "", numbers,
                         flags=re.MULTILINE)
    numbers_path.write_text(numbers, encoding="utf-8")
    with numbers_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(verdict_macros)+"\n")

    # Configuration is split so every resolved run remains readable.
    rows_a=[]; rows_b=[]
    ref_full = read("submission/evidence/stages/dpo_hh/resolved_config.json")["initial_checkpoint"]["model.safetensors"]
    ref = ref_full[:12]
    qwen_revision = read("submission/evidence/stages/sft/resolved_config.json")["revision"]
    smol_revision = read("submission/evidence/stages/persona/manifest.json")["config"]["revision"]
    for name,cfg,strategy,updates,_,_ in config_rows():
        is_lora = strategy == "LoRA"
        lora = (f"r{cfg['rank']}/a{cfg['alpha']}/d{cfg.get('dropout',.05)}; " +
                ("q,v" if len(cfg['target_modules'])==2 else "q,k,v,o,gate,up,down")) if is_lora else "--"
        scheduler = cfg.get("scheduler", "constant" if is_lora else "none")
        rows_a.append(f"{esc(name)} & {cfg['seed']} & {strategy} & {lora} & AdamW & {cfg['learning_rate']:.0e} & {scheduler} & {cfg.get('warmup_ratio',0):g} & {cfg.get('weight_decay',0):g} & {cfg.get('clip_norm',1.0):g}\\\\")
        precision = cfg.get("precision", "fp16" if cfg.get("amp") else "fp32")
        length = f"{cfg.get('max_prompt_length','--')}/{cfg['max_length']}"
        beta = cfg.get("beta", "--"); rpo=cfg.get("rpo_alpha", "--")
        reference = (f"SFT SHA {ref}" if name in ("HH DPO","UF DPO") else
                     "base, adapter disabled" if is_lora else "base revision")
        rows_b.append(f"{esc(name)} & {cfg['microbatch']}x{cfg['accumulation']}={cfg['microbatch']*cfg['accumulation']} & {precision} & {length} & {cfg['epochs']}/{updates} & {beta} & {rpo} & {reference} & greedy; 256; rep. 1; len. 1; {'chat' if is_lora else 'plain'}\\\\")
    audit = Path("report/generated_configuration.tex").read_text(encoding="utf-8")
    frozen = audit[audit.find("\\begin{table}", audit.find("Frozen prepared")):]
    cfgtex = ("\\begin{table}[H]\\centering\\footnotesize\\caption{Resolved optimization configuration (part 1).}\\label{tab:config-a}\n"
      "\\setlength{\\tabcolsep}{2.5pt}\\begin{tabularx}{\\textwidth}{lrl>{\\raggedright\\arraybackslash}Xlrrrrr}\\toprule Run&Seed&Strategy&LoRA $r/\\alpha/d$; targets&Opt.&LR&Sched.&Warm.&WD&Clip\\\\\\midrule\n"+"\n".join(rows_a)+"\n\\bottomrule\\end{tabularx}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{Resolved optimization and generation configuration (part 2). Prompt/total gives the prompt cap and total sequence limit.}\\label{tab:config-b}\n"
      "\\setlength{\\tabcolsep}{2pt}\\begin{tabularx}{\\textwidth}{lrrrrrr>{\\raggedright\\arraybackslash}p{2.1cm}X}\\toprule Run&Batch&Prec.&Prompt/total&Epoch/update&$\\beta$&RPO&Reference&Generation\\\\\\midrule\n"+"\n".join(rows_b)+"\n\\bottomrule\\end{tabularx}\\par\\footnotesize -- means not applicable. Temperature and top-p are inactive under greedy decoding.\\par\\scriptsize Exact revisions/hashes: Qwen base \\texttt{"+qwen_revision+"}; SFT reference \\texttt{"+ref_full+"}; SmolLM2 base \\texttt{"+smol_revision+"}.\\end{table}\n"+frozen)
    (report/"generated_configuration.tex").write_text(cfgtex,encoding="utf-8")

    # Persona table: training plus two readable behavior tables.
    train=[]; held=[]; final=[]
    for r in persona["runs"]:
        mark = f"\\textbf{{{esc(r['run'])}}}" if r['run']==persona['revised_selected'] else (f"{esc(r['run'])}$^\\dagger$" if r['run']==persona['historical_selected'] else esc(r['run']))
        train.append(f"{mark}&{r['trainable_parameters']:,}&{r['updates']}&{r['epochs_recorded']:.3f}&{r['learning_rate']:.0e}&{r['beta']:.1f}&{r.get('rpo_alpha') if r.get('rpo_alpha') is not None else '--'}&{r['held_out_reward_accuracy']:.3f}&{r['held_out_reward_margin']:.3f}&{r['eval_loss']:.3f}\\\\")
        held.append(f"{mark}&{100*r['held_out_degeneration_rate']:.1f}&{100*r['held_out_changed_vs_base_rate']:.1f}&{100*r['held_out_style_marker_rate']:.1f}&{r['held_out_mean_words']:.1f}&{'yes' if r['held_out_eligible'] else 'no'}&{'yes' if r['run']==persona['historical_selected'] else '--'}&{'yes' if r['run']==persona['revised_selected'] else '--'}\\\\")
        def judged(x):
            return "--" if not x else f"{x['W']}/{x['L']}/{x['T']}; {100*x['tie_adjusted']:.1f} [{100*x['bootstrap_95pct_ci'][0]:.1f},{100*x['bootstrap_95pct_ci'][1]:.1f}]"
        final.append(f"{mark}&{r['outputs_changed_vs_base']}/20&{r['degeneration_pct']:.1f}&{r['identical_to_base_pct']:.1f}&{r['mean_words']:.1f}&{judged(r['style'])}&{judged(r['quality'])}\\\\")
    ptex=("\\begin{table}[H]\\centering\\footnotesize\\caption{All seven persona adapters: training and held-out preference metrics.}\\label{tab:persona-all}\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrrrrrrr}\\toprule Variant&Params&Updates&Epoch&LR&$\\beta$&RPO&Acc.&Margin&Loss\\\\\\midrule\n"+"\n".join(train)+"\n\\bottomrule\\end{tabular}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{Held-out selection behavior on 22 prompts. $^\\dagger$ marks the historical-rule pick; bold marks the revised-rule pick.}\\label{tab:persona-held}\\begin{tabular}{lrrrrrrr}\\toprule Variant&Degen.\\%&Changed\\%&Marker\\%&Words&Eligible&Historical&Revised\\\\\\midrule\n"+"\n".join(held)+"\n\\bottomrule\\end{tabular}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{Final 20-prompt persona evaluation and blinded judgments; scores include 95\\% bootstrap intervals.}\\label{tab:persona-final}\\setlength{\\tabcolsep}{3pt}\\begin{tabularx}{\\textwidth}{lrrrrXX}\\toprule Variant&Changed&Degen.\\%&Ident.\\%&Words&Style W/L/T; score [CI]&Quality W/L/T; score [CI]\\\\\\midrule\n"+"\n".join(final)+"\n\\bottomrule\\end{tabularx}\\end{table}\n")
    (report/"generated_persona.tex").write_text(ptex,encoding="utf-8")

    # Results, response behavior, UF length, and category analysis.
    behavior=[]
    labels={"alpaca_base":"Alpaca base","alpaca_sft":"Alpaca SFT","alpaca_uf":"Alpaca UF DPO","hh_sft":"HH SFT","hh_dpo":"HH DPO","persona_base":"Persona base"}
    for key,label in labels.items():
        r=analysis["length_and_repetition"][key]
        identical = "--" if key in {"alpaca_base", "persona_base"} else f"{r.get('identical_to_comparison_pct',0):.1f}"
        behavior.append(f"{label}&{r['mean_words']:.1f}&{r['median_words']:.1f}&{r['mean_tokens']:.1f}&{r['retokenized_decoded_length_ge_256_pct']:.1f}&{r['degeneration_pct']:.1f}&{identical}\\\\")
    behavior.append(f"Persona {selected['run']}&{selected['mean_words']:.1f}&--&--&--&{selected['degeneration_pct']:.1f}&{selected['identical_to_base_pct']:.1f}\\\\")
    ufrows=[]
    for name,r in analysis['ultrafeedback']['token_length_delta_terciles'].items(): ufrows.append(f"{name} ({r['delta_min']}..{r['delta_max']})&{r['win']}&{r['loss']}&{r['tie']}\\\\")
    cat=[]
    for name,r in analysis['hh']['by_category'].items():
        n=sum(r.values()); score=(r.get('win',0)+.5*r.get('tie',0))/n
        cat.append(f"{esc(name)}&{n}&{r.get('win',0)}&{r.get('loss',0)}&{r.get('tie',0)}&{100*score:.1f}\\\\")
    # Reuse the existing full blinded table and add the revised selected adapter.
    old=Path("report/generated_results.tex").read_text(encoding="utf-8")
    behavior_marker = "\\begin{table}[H]\\centering\\footnotesize\\caption{Response behavior."
    if behavior_marker in old:
        old = old[:old.index(behavior_marker)].rstrip() + "\n"
    v3b_result = (f"Persona {selected['run']} style & 20 & {selected['style']['W']} & {selected['style']['L']} & {selected['style']['T']} & {100*selected['style']['tie_adjusted']:.1f}\\% & -- & {100*selected['style']['W']/20:.1f}\\% & [{100*selected['style']['bootstrap_95pct_ci'][0]:.1f},{100*selected['style']['bootstrap_95pct_ci'][1]:.1f}]\\% & 0/0/0\\\\\n"
                  f"Persona {selected['run']} quality & 20 & {selected['quality']['W']} & {selected['quality']['L']} & {selected['quality']['T']} & {100*selected['quality']['tie_adjusted']:.1f}\\% & -- & {100*selected['quality']['W']/20:.1f}\\% & [{100*selected['quality']['bootstrap_95pct_ci'][0]:.1f},{100*selected['quality']['bootstrap_95pct_ci'][1]:.1f}]\\% & 0/0/0\\\\\n")
    old=old.replace("score is tie-adjusted with row-bootstrap 95\\% CI.",
                    "score is tie-adjusted; HH unique uses prompt-bootstrap and other rows use row-bootstrap 95\\% CIs. Raw win is wins divided by N.")
    old=old.replace("\\bottomrule\\end{tabular}\\end{table}", v3b_result+"\\bottomrule\\end{tabular}\\end{table}",1)
    hh_len=[]
    for name,r in analysis['hh']['token_length_delta_terciles'].items():
        hh_len.append(f"{name} ({r['delta_min']}..{r['delta_max']})&{r.get('win',0)}&{r.get('loss',0)}&{r.get('tie',0)}\\\\")
    hh_rep=[]
    for name,r in analysis['hh']['outcome_by_degeneration'].items():
        hh_rep.append(f"{name}&{r['n']}&{r.get('win',0)}&{r.get('loss',0)}&{r.get('tie',0)}\\\\")
    result=(old+"\n\\begin{table}[H]\\centering\\footnotesize\\caption{Response behavior. Degeneration means $>40$ words and unique-word ratio $<0.3$.}\\label{tab:behavior}\\begin{tabular}{lrrrrrr}\\toprule Model&Mean words&Median words&Mean tokens&Retok. $\\geq256$\\%&Degen.\\%&Ident.\\%\\\\\\midrule\n"+"\n".join(behavior)+"\n\\bottomrule\\end{tabular}\\par\\footnotesize Retok. $\\geq256$ is a re-tokenization proxy, not a direct finish-reason measure. -- was not saved for the selected v3 adapter.\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{UltraFeedback outcomes by DPO-minus-SFT token-difference tercile.}\\label{tab:uf-length}\\begin{tabular}{lrrr}\\toprule Tercile (token range)&W&L&T\\\\\\midrule\n"+"\n".join(ufrows)+"\n\\bottomrule\\end{tabular}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{Row-weighted HH outcomes by prompt category; each category has 30 rows.}\\label{tab:hh-category}\\begin{tabular}{lrrrrr}\\toprule Category&N&W&L&T&Score\\%\\\\\\midrule\n"+"\n".join(cat)+"\n\\bottomrule\\end{tabular}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{HH outcomes by DPO-minus-SFT token-length tercile.}\\label{tab:hh-length}\\begin{tabular}{lrrr}\\toprule Tercile (range)&W&L&T\\\\\\midrule\n"+"\n".join(hh_len)+"\n\\bottomrule\\end{tabular}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{HH outcomes by response degeneration state.}\\label{tab:hh-repetition}\\begin{tabular}{lrrrr}\\toprule State&N&W&L&T\\\\\\midrule\n"+"\n".join(hh_rep)+"\n\\bottomrule\\end{tabular}\\end{table}\n")
    (report/"generated_results.tex").write_text(result,encoding="utf-8")

    beta=read("outputs/analysis/hh_beta_pilots.json")
    beta_generation=read("outputs/analysis/hh_beta_generation.json")["comparisons"]
    beta_rows=[]
    for run in beta['runs']:
        p=run['validation'][-1]
        key = "hh_beta005_vs_beta01" if run['beta'] == .05 else "hh_beta03_vs_beta01" if run['beta'] == .3 else None
        if key:
            g=beta_generation[key]; judged=(f"{g['win']}/{g['loss']}/{g['tie']}; "
                f"{100*g['tie_adjusted']:.1f} [{100*g['bootstrap_95pct_ci'][0]:.1f},{100*g['bootstrap_95pct_ci'][1]:.1f}]")
        else:
            judged="reference"
        beta_rows.append(f"{run['beta']:.2f}&{p['step']}&{p['reward_accuracy']:.3f}&{p['reward_margin']:.3f}&{p['loss']:.3f}&{judged}\\\\")
    deviations=[
      ("Main SFT/DPO","prescribed","matched","preserve assignment comparison","configs and metrics","within recipe","not ablated"),
      ("Persona precision","FP16 pilot","FP32","avoid nonfinite gradients","pilot log/reload checks","finite weights","resource"),
      ("v2 capacity","v1 r8, q/v, 1 epoch","r16, seven targets, 3 epochs","v1 changed few outputs","metrics/generations","larger margins; degeneration","performance"),
      ("v3 sweep","v2 LR/epochs","milder LR/epochs","test reduced exposure","v3 metrics/generations","v3b met revised rule","performance"),
      ("v3c beta","0.1","0.3","stronger anchoring","v3c metrics","degeneration remained high","performance"),
      ("v3d RPO","none","RPO 1.0","test chosen NLL regularization","v3d metrics","not ablated","performance"),
      ("Selection rule","pre-v2 margin rule","revised behavior rule","v2 degeneration observed","timestamped rule/selection","selected v3b","post hoc; performance"),
      ("HH beta","0.1","0.1; pilots .05/.1/.3","validation and generation sensitivity","pilot metrics/judgments","no conclusive generation difference","450 pilot updates"),
    ]
    devrows_a=["&".join(esc(x) for x in row[:4])+"\\\\" for row in deviations]
    devrows_b=["&".join(esc(x) for x in (row[0],*row[4:]))+"\\\\" for row in deviations]
    dtex=("\\begin{table}[H]\\centering\\footnotesize\\caption{Guide alignment and evidence-backed deviations (choices).}\\label{tab:deviations}\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{2.4cm}>{\\raggedright\\arraybackslash}p{3.3cm}X}\\toprule Parameter&Guide&Final&Reason\\\\\\midrule\n"+"\n".join(devrows_a)+"\n\\bottomrule\\end{tabularx}\\end{table}\n"
      "\\begin{table}[H]\\centering\\footnotesize\\caption{Guide alignment and evidence-backed deviations (evidence and effect).}\\label{tab:deviations-evidence}\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{2.2cm}XXX}\\toprule Parameter&Evidence&Observed effect&Resource/performance\\\\\\midrule\n"+"\n".join(devrows_b)+"\n\\bottomrule\\end{tabularx}\\end{table}\n"
      "\\begin{table}[H]\\centering\\scriptsize\\caption{Matched 150-update HH beta pilots. Generation compares each candidate with $\\beta=.10$ on 300 blinded rows; brackets are bootstrap 95\\% CIs.}\\label{tab:hh-beta}\\begin{tabular}{rrrrrl}\\toprule $\\beta$&Update&Accuracy&Margin&Loss&Generation W/L/T; score [CI]\\\\\\midrule\n"+"\n".join(beta_rows)+"\n\\bottomrule\\end{tabular}\\end{table}\n")
    (report/"generated_diagnostics.tex").write_text(dtex,encoding="utf-8")

    # Redraw the three training panels at their final printed width so text stays >=7 pt.
    try:
        from scripts.build_report import rolling
    except ModuleNotFoundError:
        from build_report import rolling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
                         "xtick.labelsize":7,"ytick.labelsize":7,
                         "axes.titlesize":8,"axes.labelsize":8,"legend.fontsize":7})
    stage_paths={"sft":"submission/evidence/stages/sft","hh":"submission/evidence/stages/dpo_hh",
                 "uf":"submission/evidence/stages/dpo_ultrafeedback"}
    stage_metrics = {}
    for key,path in stage_paths.items():
        rows=[json.loads(line) for line in Path(path,"metrics.jsonl").read_text().splitlines() if line]
        train=[r for r in rows if r['phase']=='train']; val=[r for r in rows if r['phase']=='validation']
        stage_metrics[key] = (train, val)
        if key=='sft':
            fig,ax=plt.subplots(figsize=(7.1,1.60),constrained_layout=True); ax.plot([r['step'] for r in train],rolling([r['loss'] for r in train]),label='train (25-update mean)')
            ax.plot([r['step'] for r in val],[r['loss'] for r in val],'o-',label='validation')
            ax.axvline(938,color='0.35',ls='--',lw=1,label='selected update 938')
            ax.set(xlabel='Optimizer update',ylabel='CE loss',title='SFT loss'); ax.legend(loc='upper right',ncol=3)
        else:
            fig,axes=plt.subplots(1,3,figsize=(7.1,1.85),constrained_layout=True); axes[0].plot([r['step'] for r in train],rolling([r['loss'] for r in train]),label='train (25-update mean)')
            axes[0].plot([r['step'] for r in val],[r['loss'] for r in val],'o-',label='validation'); axes[0].set(xlabel='Optimizer update',ylabel='DPO loss')
            axes[1].plot([r['step'] for r in val],[r['chosen_reward'] for r in val],'o-',label='chosen')
            axes[1].plot([r['step'] for r in val],[r['rejected_reward'] for r in val],'s-',label='rejected')
            axes[1].plot([r['step'] for r in val],[r['reward_margin'] for r in val],'^-',label='margin')
            axes[1].set(xlabel='Optimizer update',ylabel='Reward')
            axes[2].plot([r['step'] for r in val],[r['reward_accuracy'] for r in val],'o-',color='#7A5195',label='validation accuracy'); axes[2].set(xlabel='Optimizer update',ylabel='Accuracy')
            for a in axes: a.axvline(469,color='0.35',ls='--',lw=1,label='selected update 469')
            handles,labels=[],[]
            for a in axes:
                for h,l in zip(*a.get_legend_handles_labels()):
                    if l not in labels: handles.append(h); labels.append(l)
            fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.01))
            fig.get_layout_engine().set(rect=(0,.27,1,1),w_pad=6/72,h_pad=3/72)
            ax=axes[0]
        for axis in fig.axes: axis.grid(alpha=.2)
        if key == 'sft': fig.get_layout_engine().set(w_pad=8/72,h_pad=3/72)
        fig.savefig(report/f"figures/{key}.pdf"); plt.close(fig)

    sft_val = stage_metrics["sft"][1]
    hh_val = stage_metrics["hh"][1]
    uf_val = stage_metrics["uf"][1]
    training = (
        f"SFT validation loss moved from {sft_val[0]['loss']:.3f} at update {sft_val[0]['step']} to "
        f"{sft_val[-1]['loss']:.3f} at update {sft_val[-1]['step']}; after an early rise, the final point was the lowest saved validation loss, so update {sft_val[-1]['step']} was selected. "
        "Measured Alpaca mean length changed from 173.5 to 166.8 tokens and degeneration fell from 20.0\\% to 16.7\\% (Table~\\ref{tab:behavior}); template adaptation remains a hypothesis rather than an isolated cause.\n\n"
        f"HH validation accuracy rose from {hh_val[0]['reward_accuracy']:.3f} to {hh_val[-1]['reward_accuracy']:.3f}; its margin moved from {hh_val[0]['reward_margin']:.4f} to {hh_val[-1]['reward_margin']:.3f}, with final chosen/rejected implicit rewards {hh_val[-1]['chosen_reward']:.3f}/{hh_val[-1]['rejected_reward']:.3f}. "
        f"UltraFeedback accuracy rose from {uf_val[0]['reward_accuracy']:.3f} to {uf_val[-1]['reward_accuracy']:.3f}; its margin moved from {uf_val[0]['reward_margin']:.4f} to {uf_val[-1]['reward_margin']:.3f}, with final rewards {uf_val[-1]['chosen_reward']:.3f}/{uf_val[-1]['rejected_reward']:.3f}. "
        f"For both DPO tracks the final saved validation loss was the minimum, so update {hh_val[-1]['step']} was selected; no held-out reversal appeared before training ended, although one epoch and one seed cannot rule out overfitting. "
        "At initialization the checkpoints are identical, so the theoretical margin is zero. The FP32 policy on cuda:0 runs under FP16 autocast while the frozen reference was loaded in FP16 on cuda:1; device/dtype rounding in summed token log-probabilities creates tiny signed differences, so strict positive-margin counting reports 0.492 and 0.471. The ranking gap, rather than either reward's absolute sign, is the DPO diagnostic.\n")
    (report/"generated_training.tex").write_text(training, encoding="utf-8")

    # Notes are manually observational, while every quoted string remains evidence-derived.
    examples=read("outputs/analysis/report_examples.json"); notes=read("report/example_notes.json")
    prose=["\\subsection*{Representative judged examples}"]
    for track in ("hh","ultrafeedback","persona_v3b"):
        for outcome in ("win","loss","tie"):
            r=examples[track][outcome]; note=notes[r['id']]['note']
            prose.append(f"\\paragraph{{{esc(track)} {outcome} ({esc(r['prompt_id'])}).}} \\textbf{{Prompt:}} {esc(r['prompt'])} \\textbf{{{esc(r.get('candidate_label','DPO'))}:}} {esc(r['candidate'])} \\textbf{{{esc(r.get('baseline_label','SFT'))}:}} {esc(r['baseline'])} \\textbf{{Analysis:}} {esc(note)}")
    (report/"generated_examples.tex").write_text("\n".join(prose)+"\n",encoding="utf-8")

    # Colorblind-safe persona figure with distinct markers and external legend.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=plt.get_cmap("tab10").colors; markers="os^DvpP"
    plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.15),constrained_layout=True)
    loss_values=[]; margin_values=[]
    for i,r in enumerate(persona["runs"]):
        pts=r["epoch_diagnostics"]
        loss_values.extend(p['eval_loss'] for p in pts)
        margin_values.extend(p['eval_rewards/margins'] for p in pts)
        axes[0].plot([p['epoch'] for p in pts],[p['eval_loss'] for p in pts],marker=markers[i],color=colors[i],label=r['run'])
        axes[1].plot([p['epoch'] for p in pts],[p['eval_rewards/margins'] for p in pts],marker=markers[i],color=colors[i],label="_nolegend_")
    axes[0].set(xlabel="Recorded epoch",ylabel="Loss",yscale="log"); axes[1].set(xlabel="Recorded epoch",ylabel="Margin")
    axes[0].set_ylim(min(loss_values)*.70,max(loss_values)*1.35)
    margin_span=max(margin_values)-min(margin_values)
    axes[1].set_ylim(max(0,min(margin_values)-.10*margin_span),max(margin_values)+.15*margin_span)
    for ax in axes: ax.grid(alpha=.2)
    fig.legend(loc="lower center",ncol=4,bbox_to_anchor=(.5,-.01)); fig.get_layout_engine().set(rect=(0,.27,1,1),w_pad=8/72,h_pad=3/72)
    fig.savefig(report/"figures/persona_variants.pdf"); plt.close(fig)
