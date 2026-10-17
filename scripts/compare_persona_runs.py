"""Build the measured persona comparison; add v3 rows when their evidence exists."""
import argparse, json, math, random, re
from collections import Counter
from pathlib import Path


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
def words(text): return re.findall(r"\b[a-z][a-z'-]{2,}\b", text.lower())

def ci(counts, seed=42, samples=10000):
    vals=[1]*counts['W']+[0]*counts['L']+[.5]*counts['T']; rng=random.Random(seed); n=len(vals)
    scores=sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return [scores[int(.025*samples)],scores[int(.975*samples)-1]]

def judgments(root, track):
    row=read_json(Path(root)/f'{track}_summary.json')[track]
    c={k:row[k] for k in ('W','L','T')}; c['tie_adjusted']=row['tie_adjusted_score']; c['bootstrap_95pct_ci']=ci(c)
    c['unknown']=row['U']; c['failures']=row['F']; return c

def markers(data):
    rows=read_json(data); chosen=Counter(); rejected=Counter()
    stop={'this','that','with','from','your','have','will','would','could','should','about','there','their','they','them','then','than','into','when','what','where','which','while','just','also','some','more','very','here','sure'}
    for r in rows:
        chosen.update(set(words(r['chosen']))-stop); rejected.update(set(words(r['rejected']))-stop)
    return sorted(w for w,c in chosen.items() if c>=5 and (c+1)/(rejected[w]+1)>=2)[:40]

def generation_stats(base_path, gen_path, lexicon):
    base=read_json(base_path); gen=read_json(gen_path); assert [r['id'] for r in base]==[r['id'] for r in gen]
    lengths=[len(words(r['output'])) for r in gen]
    deg=[n>40 and len(set(words(r['output'])))/n<.3 for r,n in zip(gen,lengths)]
    return {'outputs_changed_vs_base':sum(a['output']!=b['output'] for a,b in zip(base,gen)),
            'n':len(gen),'mean_words':sum(lengths)/len(lengths),
            'degeneration_pct':100*sum(deg)/len(deg),
            'identical_to_base_pct':100*sum(a['output']==b['output'] for a,b in zip(base,gen))/len(gen),
            'style_marker_rate_pct':100*sum(any(w in set(words(r['output'])) for w in lexicon) for r in gen)/len(gen)}

def metric_row(name,stage,base,gen,evaluation,lexicon):
    manifest=read_json(Path(stage)/'manifest.json'); metrics=read_json(Path(stage)/'metrics.json'); cfg=manifest['config']; final=metrics['evaluation']
    train=[x for x in metrics['history'] if 'loss' in x]; evals=[]; seen=set()
    for x in metrics['history']:
        if 'eval_loss' in x and x.get('step') not in seen: evals.append({k:x.get(k) for k in ('step','epoch','eval_loss','eval_rewards/accuracies','eval_rewards/margins','eval_rewards/chosen','eval_rewards/rejected')});seen.add(x.get('step'))
    per_epoch=[]
    for x in evals:
        upto=[t for t in train if t['step']<=x['step']]
        last=upto[-1]; per_epoch.append({'step':x['step'],'epoch':x['epoch'],'train_reward_accuracy':last['rewards/accuracies'],'train_reward_margin':last['rewards/margins'],**x})
    return {'run':name,'selected':name=='v2_lr2e4','trainable_parameters':manifest['trainable'],'updates':max(x['step'] for x in train),
      'epochs_recorded':max(x['epoch'] for x in train),'learning_rate':cfg['learning_rate'],
      'beta':cfg['beta'],'rpo_alpha':cfg.get('rpo_alpha'),'rank':cfg['rank'],'alpha':cfg['alpha'],
      'dropout':cfg['dropout'],'target_modules':cfg['target_modules'],
      'held_out_reward_accuracy':final['eval_rewards/accuracies'],'held_out_reward_margin':final['eval_rewards/margins'],'eval_loss':final['eval_loss'],
      'held_out_chosen_reward':final['eval_rewards/chosen'],'held_out_rejected_reward':final['eval_rewards/rejected'],
      'epoch_diagnostics':per_epoch,**generation_stats(base,gen,lexicon),
      'style':judgments(evaluation,'persona_style') if evaluation else None,
      'quality':judgments(evaluation,'persona_quality') if evaluation else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='outputs/analysis/persona_comparison.json'); args=ap.parse_args()
    lex=markers('data/persona_lora/ramsay_style_preferences.json'); base='outputs/persona-generation-v1/outputs/generations/persona_base.json'
    rows=[metric_row('v1','outputs/persona_v2/outputs/persona_v1_reproduction',base,'outputs/persona-generation-v1/outputs/generations/persona_lora.json','outputs/evaluation',lex),
      metric_row('v2_lr2e4','outputs/persona_v2/outputs/persona_v2',base,'outputs/persona_v2_generation/outputs/generations/persona_lora_v2.json','outputs/persona_v2_evaluation/persona_lora_v2',lex),
      metric_row('v2_lr1e4','outputs/persona_v2/outputs/persona_v2_lr1e4',base,'outputs/persona_v2_generation/outputs/generations/persona_lora_v2_lr1e4.json','outputs/persona_v2_evaluation/persona_lora_v2_lr1e4',lex)]
    selection_path=Path('outputs/analysis/persona_selection_v2.json')
    revised=read_json(selection_path) if selection_path.exists() else None
    v3root=Path('outputs/persona_v3/outputs')
    if v3root.exists():
        for name in ('v3a','v3b','v3c','v3d'):
            rows.append(metric_row(name,v3root/name,base,v3root/'generations'/f'persona_{name}.json',None,lex))
    v3_judging=Path('outputs/analysis/persona_v3_judging_summary.json')
    if v3_judging.exists():
        judged=read_json(v3_judging)['tracks']
        for row in rows:
            if row['run'].startswith('v3'):
                for rubric in ('style','quality'):
                    source=judged[f"persona_{row['run']}_{rubric}"]
                    row[rubric]={'W':source['W'],'L':source['L'],'T':source['T'],
                                 'tie_adjusted':source['tie_adjusted_score'],
                                 'bootstrap_95pct_ci':source['bootstrap_95pct_ci'],
                                 'unknown':source['U'],'failures':source['F']}
    if revised:
        heldout={row['name']:row for row in revised['adapters']}
        for row in rows:
            if row['run'] in heldout:
                row.update({f"held_out_{key}":value for key,value in heldout[row['run']].items()
                            if key in ('degeneration_rate','changed_vs_base_rate','style_marker_rate','mean_words','eligible')})
    out={'historical_selection_rule':'highest held-out reward accuracy; ties by larger held-out margin, then lower eval loss',
         'historical_selected':'v2_lr2e4','selection_rule':'highest held-out reward accuracy; ties by larger held-out margin, then lower eval loss',
         'selected':'v2_lr2e4','revised_selection_rule':'docs/persona_selection_rule_v2.md',
         'revised_selected':revised['selected'] if revised else None,
         'style_marker_definition':'lowercase word types occurring in >=5 chosen training responses and with add-one chosen/rejected document-frequency ratio >=2; rate is outputs containing >=1 marker','style_markers':lex,'runs':rows}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
