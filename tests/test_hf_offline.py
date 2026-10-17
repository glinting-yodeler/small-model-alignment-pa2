"""Real Transformers/TRL integration on random tiny local weights; no downloads."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from alignment.io import write_json, read_json, read_jsonl


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
    vocab={w:i for i,w in enumerate(['<pad>','<eos>','<unk>','<bos>','user','assistant','q','a','b','good','bad','Instruction','Response',':','#'])}
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'))
    backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='<eos>',bos_token='<bos>',unk_token='<unk>')
    tok.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    cfg=LlamaConfig(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
                    bos_token_id=3,eos_token_id=1,pad_token_id=0)
    model=LlamaForCausalLM(cfg)
    root=tmp_path/'tiny'; model.save_pretrained(root);tok.save_pretrained(root)
    return root


def test_generation_reload_resume_left_padding_and_identity(tiny_checkpoint,tmp_path):
    from alignment.generation import generate
    prompts=[dict(id='1',instruction='q',dataset='tiny'),dict(id='2',instruction='q q q',dataset='tiny')]
    write_json(tmp_path/'prompts.json',prompts)
    cfg=dict(model=str(tiny_checkpoint),prompts=str(tmp_path/'prompts.json'),output=str(tmp_path/'gen.json'),
             generator='tiny',device='cpu',batch_size=2,max_new_tokens=3,format='plain')
    generate(cfg)
    first=read_json(tmp_path/'gen.json')
    assert [r['id'] for r in first]==['1','2']
    assert all(len(r['output'].split())<=3 for r in first)
    generate(cfg)
    assert read_json(tmp_path/'gen.json')==first and len(read_jsonl(tmp_path/'gen.jsonl'))==2
    with pytest.raises(ValueError,match='identity'):generate(dict(cfg,max_new_tokens=4))


def test_persona_trl_lora_step_save_and_reload(tiny_checkpoint,tmp_path,monkeypatch):
    import huggingface_hub
    from alignment.persona import run_persona
    monkeypatch.setattr(huggingface_hub,'model_info',lambda *a,**k:SimpleNamespace(sha=None))
    rows=[dict(prompt='q '+str(i),chosen='good a',rejected='bad b') for i in range(6)]
    write_json(tmp_path/'data.json',rows)
    write_json(tmp_path/'eval.json',[dict(instruction='unseen prompt')])
    cfg=dict(model=str(tiny_checkpoint),output=str(tmp_path/'persona'),data=str(tmp_path/'data.json'),
             evaluation_prompts=str(tmp_path/'eval.json'),validation_size=2,seed=42,rank=2,alpha=4,
             dropout=0.,target_modules=['q_proj','v_proj'],learning_rate=.001,beta=.1,microbatch=1,
             accumulation=1,max_length=32,epochs=1,max_updates=1)
    run_persona(cfg)
    manifest=read_json(tmp_path/'persona/manifest.json')
    assert 0<manifest['fraction']<1
    assert (tmp_path/'persona/adapter/adapter_model.safetensors').exists()
    assert read_json(tmp_path/'persona/reload_verified.json')['ok']


def test_custom_hf_sft_and_both_dpo_branches_reload(tiny_checkpoint,tmp_path,monkeypatch):
    import huggingface_hub
    from alignment.training import run_training
    monkeypatch.setattr(huggingface_hub,'model_info',lambda *a,**k:SimpleNamespace(sha=None))
    rows=[dict(id=str(i),prompt='q '+str(i),response='good a',chosen='good a',rejected='bad b') for i in range(4)]
    write_json(tmp_path/'train.json',rows[:3]);write_json(tmp_path/'val.json',rows[3:])
    cfg=dict(track='sft',model=str(tiny_checkpoint),output=str(tmp_path/'sft'),device='cpu',seed=42,
             max_length=32,microbatch=1,accumulation=2,learning_rate=.001,weight_decay=0.,
             warmup_ratio=0.,scheduler='none',clip_norm=1.,epochs=1,amp=False,eval_every=1,
             save_every=1,max_updates=1,pilot_audit=True,gradient_checkpointing=True,
             train_data=str(tmp_path/'train.json'),validation_data=str(tmp_path/'val.json'))
    run_training(cfg)
    for name in ['dpo_hh','dpo_ultrafeedback']:
        dpo=dict(cfg,track=name,model=str(tmp_path/'sft/best'),output=str(tmp_path/name),
                 beta=.1,reference_device='cpu',max_prompt_length=16)
        run_training(dpo)
        audit=read_json(tmp_path/name/'pilot_audit.json')
        assert audit['reference_unchanged'] and audit['checkpoint_reload']
    hh=read_json(tmp_path/'dpo_hh/resolved_config.json')['initial_checkpoint']
    uf=read_json(tmp_path/'dpo_ultrafeedback/resolved_config.json')['initial_checkpoint']
    assert hh==uf
