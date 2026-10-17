import copy
import json
from pathlib import Path
import pytest
import torch
from alignment.data import parse_sft, parse_preference, select_splits, normalize
from alignment.io import read_jsonl, read_json, load_config
from alignment.inputs import find_prepared_data
from alignment.judging import judge, judge_prompt, parse_decisions
from alignment.pairwise import build_pairs, summarize
from alignment.persona import require_finite_metrics
from alignment.training import train, encode_records, validate
from tests.toy_lm import ToyCausalLM, ToyTokenizer
from scripts import prepare_kaggle_stage


def test_committed_learning_rates_are_numeric():
    for name in ("sft", "dpo_hh", "dpo_ultrafeedback", "persona_lora"):
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml")
        assert isinstance(config["learning_rate"], float)
        assert 0 < config["learning_rate"] < 1


def test_persona_training_rejects_nonfinite_gradient_metrics():
    require_finite_metrics({"eval_loss": 0.69}, [{"grad_norm": 0.2}])
    with pytest.raises(FloatingPointError, match="grad_norm"):
        require_finite_metrics({"eval_loss": 0.69}, [{"grad_norm": float("nan")}])
    with pytest.raises(FloatingPointError, match="eval_loss"):
        require_finite_metrics({"eval_loss": float("inf")}, [])


def test_judge_parses_exact_fenced_json_without_accepting_prose():
    fenced = '```json\n[{"id":"p1","decision":"B"}]\n```'
    assert parse_decisions(fenced, {"p1"}) == {"p1": "B"}
    assert parse_decisions('```\n[{"id":"p1","decision":"tie"}]\n```', {"p1"}) == {"p1": "tie"}
    with pytest.raises(json.JSONDecodeError):
        parse_decisions('Here is my answer: ' + fenced, {"p1"})


def test_kaggle_stage_bootstrap_imports_extracted_source(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "alignment").mkdir()
    (root / "alignment" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "requirements-kaggle.txt", "submission_adapter.py"):
        (root / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(prepare_kaggle_stage, "__file__", str(root / "scripts" / "prepare_kaggle_stage.py"))
    prepare_kaggle_stage.main("sft")
    wrapper = (root / "kaggle_stages" / "sft" / "stage.py").read_text(encoding="utf-8")
    compile(wrapper, "stage.py", "exec")
    assert wrapper.index("sys.path.insert(0, str(root))") < wrapper.index("from alignment.io import")


def test_kaggle_prepared_data_discovery_nested_mount_and_hashes(tmp_path):
    import hashlib
    data = tmp_path / "datasets" / "owner" / "slug"
    data.mkdir(parents=True)
    for name, content in (("manifest.json", "{}"), ("sft_train.json", "[]")):
        (data / name).write_text(content, encoding="utf-8")
    hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
              for name in ("manifest.json", "sft_train.json")}
    (data / "upload_file_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    assert find_prepared_data(tmp_path, ("sft_train.json",)) == data
    (data / "sft_train.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        find_prepared_data(tmp_path, ("sft_train.json",))


def test_multiturn_sft_extracts_first_exchange_with_system():
    msgs = [{"role": r, "content": t} for r, t in [("system","be concise"),("user","q"),("assistant","a"),("user","next"),("assistant","b")]]
    assert parse_sft({"messages": msgs}) == {"prompt": "System: be concise\n\nq", "response": "a"}


@pytest.mark.parametrize("kind", ["hh", "ultrafeedback"])
def test_preference_parsers_reject_context_mismatch_and_multiturn(kind):
    if kind == "hh":
        good = {"chosen": "\n\nHuman: q\n\nAssistant: yes", "rejected": "\n\nHuman: q\n\nAssistant: no"}
        bad = dict(good, rejected="\n\nHuman: different\n\nAssistant: no")
        multi = {k: v + "\n\nHuman: next\n\nAssistant: x" for k,v in good.items()}
    else:
        good = {k: [{"role":"user","content":"q"},{"role":"assistant","content":v}] for k,v in [("chosen","yes"),("rejected","no")]}
        bad = copy.deepcopy(good); bad["rejected"][0]["content"] = "different"
        multi = {k: v + v for k,v in good.items()}
    assert parse_preference(good, kind) == dict(prompt="q", chosen="yes", rejected="no")
    with pytest.raises(ValueError): parse_preference(bad, kind)
    with pytest.raises(ValueError): parse_preference(multi, kind)


def test_dedup_filter_before_selection_prompt_disjoint_and_counts():
    rows = [dict(prompt=f"q{i}", chosen="yes", rejected="no") for i in range(10)]
    rows += [dict(rows[0], prompt=" Q0 "), dict(rows[1], chosen="no"), {"broken": 1}]
    a,b,counts = select_splits(rows,"persona",5,3,excluded=["q9"])
    assert counts == {"raw":13,"valid":9,"evaluation_overlap":1,"duplicate_prompt":1,"identical_responses":1,"malformed_schema":1}
    assert not {normalize(r['prompt']) for r in a} & {normalize(r['prompt']) for r in b}
    assert (a,b,counts) == select_splits(rows,"persona",5,3,excluded=["q9"])
    with pytest.raises(ValueError, match="Need"): select_splits(rows,"persona",50,3)


def fixtures():
    baseline=[dict(id=str(i),instruction=f"prompt {i}",output=f"base {i}",generator="sft-secret",dataset="eval") for i in range(5)]
    candidate=[dict(r, output=f"candidate {r['id']}",generator="dpo-secret") for r in reversed(baseline)]
    return build_pairs(baseline,candidate,"hh",expected_count=5)


def test_blinded_pairs_join_by_id_and_validate():
    pairs=fixtures()
    assert pairs == fixtures()
    text=judge_prompt(pairs,"rubric")
    assert "sft-secret" not in text and "dpo-secret" not in text and "mapping" not in text
    for p in pairs:
        assert p['A'].endswith(p['prompt_id']) and p['B'].endswith(p['prompt_id'])
    with pytest.raises(ValueError): build_pairs([],[],"hh",expected_count=300)
    assert parse_decisions([{"id":"x","decision":"A"},{"id":"x","decision":"B"}],{'x'}) == {}


class Clock:
    def __init__(self): self.value=100.; self.sleeps=[]
    def now(self): return self.value
    def sleep(self,n): self.sleeps.append(n); self.value+=n


def test_partial_judging_interruption_resume_and_throttle(tmp_path):
    pairs=fixtures(); output=tmp_path/'judged.jsonl'; clock=Clock(); calls=[]
    def backend(prompt):
        calls.append(clock.now())
        if len(calls)==1:return [{"id":pairs[0]['id'],"decision":"A"}]
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        judge(pairs,output,backend,"fake",clock=clock.now,sleep=clock.sleep)
    assert len(read_jsonl(output))==1
    def resumed(prompt):
        assert pairs[0]['id'] not in prompt
        calls.append(clock.now())
        return [{"id":p['id'],"decision":"tie"} for p in pairs[1:]]
    results=judge(pairs,output,resumed,"fake",clock=clock.now,sleep=clock.sleep)
    assert len(results)==5 and all(b-a>=5 for a,b in zip(calls,calls[1:]))
    judge(pairs,output,lambda _:pytest.fail("completed pairs recalled"),"fake",clock=clock.now,sleep=clock.sleep)
    assert len(read_jsonl(output))==5
    with pytest.raises(ValueError,match="identity"):
        judge(pairs,output,resumed,"different",clock=clock.now,sleep=clock.sleep)


def test_failures_unknowns_and_denominators(tmp_path):
    pairs=fixtures(); clock=Clock()
    result=judge(pairs,tmp_path/'j.jsonl',lambda _: 'garbage',"fake",max_attempts=3,clock=clock.now,sleep=clock.sleep)
    assert not result and len(read_jsonl(tmp_path/'j.failures.jsonl'))==5
    rows=[dict(id=p['id'],pair_hash=p['hash'],decision=d,attempt=1) for p,d in zip(pairs,['A','B','tie','unknown'])]
    summary=summarize(pairs,rows)['hh']
    assert summary['N']==5 and summary['T']==1 and summary['U']==1 and summary['F']==1
    assert summary['valid_coverage']==.6
    assert summary['W']+summary['L']==2
    assert summarize(pairs,[])['hh']['decisive_win_rate'] is None


class TrainingTokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        for char in '#\n':self.vocab[char]=len(self.vocab)


def config(output,track='sft'):
    return dict(track=track,output=str(output),device='cpu',seed=42,max_length=64,microbatch=1,
                accumulation=2,learning_rate=.01,weight_decay=0.,warmup_ratio=.1,scheduler='cosine' if track=='sft' else 'none',
                clip_norm=1.,epochs=2,amp=False,eval_every=2,save_every=1,beta=.1)


@pytest.mark.parametrize('track',['sft','dpo_hh','dpo_ultrafeedback'])
def test_real_engine_exact_resume_partial_window_and_frozen_reference(tmp_path,track):
    tok=TrainingTokenizer()
    rows=[dict(id=str(i),prompt='q',response='a'*(i+1),chosen='a'*(i+1),rejected='b') for i in range(5)]
    cfg=config(tmp_path/track,track)
    initial=ToyCausalLM(tok.vocab_size)
    model=copy.deepcopy(initial); reference=copy.deepcopy(initial) if track!='sft' else None
    train(model,tok,rows,rows[:2],cfg,reference)
    expected=copy.deepcopy(model.state_dict())
    interrupted_cfg=dict(cfg,output=str(tmp_path/(track+'resume')))
    model=copy.deepcopy(initial)
    pos=train(model,tok,rows,rows[:2],interrupted_cfg,reference,stop_after=2)
    assert pos['step']==2
    model=copy.deepcopy(initial)
    pos=train(model,tok,rows,rows[:2],interrupted_cfg,reference,resume=Path(interrupted_cfg['output'])/'latest.pt')
    assert pos['step']==6
    for k,v in expected.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    metrics=read_jsonl(Path(cfg['output'])/'metrics.jsonl')
    assert [r['examples'] for r in metrics if r['phase']=='train']==[2,2,1,2,2,1]
    assert all(r['gradient_norm']>0 for r in metrics if r['phase']=='train')
    if reference:
        for k,v in reference.state_dict().items():torch.testing.assert_close(v,initial.state_dict()[k])
    with pytest.raises(ValueError,match='identity'):
        train(model,tok,rows,rows[:2],dict(interrupted_cfg,beta=.2),reference,resume=Path(interrupted_cfg['output'])/'latest.pt')


def test_token_mean_accumulation_matches_large_batch(tmp_path):
    tok=TrainingTokenizer(); rows=[dict(prompt='q',response='a'*i) for i in [1,3,7]]
    a,b=ToyCausalLM(tok.vocab_size),ToyCausalLM(tok.vocab_size)
    cfg=dict(config(tmp_path/'a'),epochs=1,accumulation=3)
    train(a,tok,rows,rows,cfg)
    train(b,tok,rows,rows,dict(cfg,output=str(tmp_path/'b'),microbatch=3,accumulation=1))
    torch.testing.assert_close(a.logit_bias,b.logit_bias,atol=1e-7,rtol=1e-6)


def test_training_rejects_empty_response_mask_and_restores_eval_mode():
    tok=TrainingTokenizer(); cfg=config('unused'); rows=[dict(prompt='q',response='a')]
    with pytest.raises(ValueError):encode_records(tok,rows,dict(cfg,max_length=1))
    model=ToyCausalLM(tok.vocab_size).train()
    validate(model,None,tok,encode_records(tok,rows,cfg),cfg)
    assert model.training


def test_preserved_public_tests_data_pdf_and_plan_hashes():
    from alignment.io import file_hash
    provenance=read_json('docs/starter_manifest.json')
    manifest=provenance['files']
    for path,sha in manifest.items():
        if path.startswith(('tests/','data/','eval_sets/','kaggle_smoke/')) or path in ('docs/assignment_manual.pdf','PA2_Codex_Implementation_Plan.md'):
            # The final handoff intentionally excludes the implementation plan;
            # preserve and hash-check it in the working repository when present.
            if path == 'PA2_Codex_Implementation_Plan.md' and not Path(path).exists():
                continue
            assert file_hash(path)==sha, path
    # requirements.txt is the one approved starter-file change.  PEFT was
    # already present in the starter; the assignment added exactly one TRL
    # line.  Preserve both hashes and assert the semantic delta explicitly.
    approved=provenance['approved_changes']['requirements.txt']
    lines=Path('requirements.txt').read_text(encoding='utf-8').splitlines()
    added=approved['exact_added_lines']
    assert [line for line in lines if line not in approved['original_lines']]==added
    assert [line for line in lines if line not in added]==approved['original_lines']
    assert {line.split('>',1)[0].split('=',1)[0] for line in lines} >= set(approved['required_packages'])
    assert file_hash('requirements.txt')==approved['current_sha256']


def test_plot_real_tiny_training_metrics(tmp_path):
    from alignment.plotting import plot_metrics
    tok=TrainingTokenizer()
    rows=[dict(prompt='q',response='a')]
    cfg=dict(config(tmp_path/'tiny_sft'),epochs=1)
    train(ToyCausalLM(tok.vocab_size),tok,rows,rows,cfg)
    plot_metrics([tmp_path/'tiny_sft/metrics.jsonl'],tmp_path/'plots')
    image=tmp_path/'plots/tiny_sft_loss.png'
    assert image.read_bytes().startswith(b'\x89PNG') and image.stat().st_size>1000
