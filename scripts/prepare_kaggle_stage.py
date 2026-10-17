"""Package one full experiment stage; called only after pilot success."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

STAGES = {
    "sft": {"kernel": "pa2-sft-full", "datasets": ["moibbi/pa2-frozen-prepared-inputs"], "kernels": []},
    "dpo_hh": {"kernel": "pa2-dpo-hh-full", "datasets": ["moibbi/pa2-frozen-prepared-inputs"], "kernels": ["moibbi/pa2-sft-full"]},
    "dpo_ultrafeedback": {"kernel": "pa2-dpo-ultrafeedback-full", "datasets": ["moibbi/pa2-frozen-prepared-inputs"], "kernels": ["moibbi/pa2-sft-full"]},
    "persona": {"kernel": "pa2-persona-full", "datasets": [], "kernels": []},
    "persona_v2": {"kernel": "pa2-persona-v2-full", "datasets": [], "kernels": []},
    "persona_v3": {"kernel": "pa2-persona-v3-full", "datasets": [],
                   "kernels": ["moibbi/pa2-persona-v2-full"]},
    "dpo_hh_pilots": {"kernel": "pa2-dpo-hh-pilots-full", "datasets": ["moibbi/pa2-frozen-prepared-inputs"], "kernels": ["moibbi/pa2-sft-full"]},
}


def main(stage):
    if stage not in STAGES:
        raise ValueError(f"Choose one of {list(STAGES)}")
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_stages" / stage
    if out.exists() and any(p.name not in {"stage.py", "kernel-metadata.json", "source_manifest.json", "__pycache__"} for p in out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite unexpected stage package contents: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files = [p for directory in ("alignment", "scripts", "configs", "tests", "data/persona_lora", "eval_sets")
             for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [root / name for name in ("pyproject.toml", "requirements-kaggle.txt", "submission_adapter.py")]
    buffer = io.BytesIO()
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(root))
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    template = '''import base64, importlib.metadata, io, json, os, subprocess, sys, zipfile
from pathlib import Path
stage = __STAGE_VALUE__
root = Path('/kaggle/working/pa2-stage-source')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
os.chdir(root)
sys.path.insert(0, str(root))
torch_version = importlib.metadata.version('torch')
Path('keep-kaggle-torch.txt').write_text('torch==' + torch_version + '\\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-kaggle.txt', '-c', 'keep-kaggle-torch.txt'], check=True)
assert importlib.metadata.version('torch') == torch_version
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'], check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))
from alignment.io import load_config, read_json, write_json
from alignment.inputs import find_prepared_data
if stage == 'persona_v2':
    for name in ('persona_lora', 'persona_lora_v2', 'persona_lora_v2_lr1e4'):
        sweep_config = load_config('configs/' + name + '.yaml')
        if name == 'persona_lora':
            sweep_config['output'] = '/kaggle/working/outputs/persona_v1_reproduction'
        resolved = Path('/kaggle/working/' + name + '_run_config.json')
        write_json(resolved, sweep_config)
        print('PERSONA_SWEEP', name, sweep_config['output'], flush=True)
        subprocess.run([sys.executable, 'scripts/train_persona.py', '--config', str(resolved)], check=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    raise SystemExit(0)
if stage == 'persona_v3':
    from alignment.data import select_splits
    from alignment.io import file_hash
    subprocess.run([sys.executable, 'scripts/env_check.py', '--output',
                    '/kaggle/working/outputs/environment.json'], check=True)
    variants = ('persona_lora_v3a', 'persona_lora_v3b', 'persona_lora_v3c', 'persona_lora_v3d')
    for name in variants:
        cfg = load_config('configs/' + name + '.yaml')
        resolved = Path('/kaggle/working/' + name + '_run_config.json')
        write_json(resolved, cfg)
        adapter = Path(cfg['output']) / 'adapter' / 'adapter_model.safetensors'
        if not adapter.exists():
            subprocess.run([sys.executable, 'scripts/train_persona.py', '--config', str(resolved)], check=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
    raw = read_json('data/persona_lora/ramsay_style_preferences.json')
    excluded = [r['instruction'] for r in read_json('eval_sets/persona_eval_20.json')]
    _, heldout, _ = select_splits(raw, 'persona', len(raw)-22, 22, 42, excluded)
    heldout_prompts = [{'id': r['id'], 'instruction': r['prompt']} for r in heldout]
    heldout_path = Path('/kaggle/working/outputs/persona_heldout_22.json')
    write_json(heldout_path, heldout_prompts)
    discovered = {}
    for adapter_file in Path('/kaggle/input').rglob('adapter_model.safetensors'):
        parent = adapter_file.parent
        manifest = parent.parent / 'manifest.json'
        if manifest.exists():
            cfg = read_json(manifest).get('config', {})
            lr, epochs = cfg.get('learning_rate'), cfg.get('epochs')
            key = 'v1' if epochs == 1 else 'v2_lr2e4' if lr == 0.0002 else 'v2_lr1e4'
            discovered[key] = str(parent)
    if set(discovered) != {'v1','v2_lr2e4','v2_lr1e4'}:
        raise ValueError('Could not uniquely discover all prior persona adapters: ' + repr(discovered))
    adapters = {**discovered, **{n.replace('persona_lora_',''): '/kaggle/working/outputs/' +
                n.replace('persona_lora_','') + '/adapter' for n in variants}}
    base_cfg = load_config('configs/generate_persona_base.yaml')
    generation_records = []
    for key, adapter_path in [('base', None), *adapters.items()]:
        cfg = dict(base_cfg)
        cfg.update(prompts=str(heldout_path), output=f'/kaggle/working/outputs/generations/heldout_{key}.json',
                   generator=f'smollm2-persona-heldout-{key}', device='cuda:0')
        if adapter_path: cfg['adapter'] = adapter_path
        else: cfg.pop('adapter', None)
        resolved = Path('/kaggle/working/generate_heldout_' + key + '.json'); write_json(resolved, cfg)
        if key.startswith('v3'):
            sanity = Path('/kaggle/working/outputs/sanity/sanity_' + key + '.json')
            subprocess.run([sys.executable, 'scripts/sanity_check_generation.py', '--config', str(resolved),
                            '--count', '3', '--output', str(sanity)], check=True)
        subprocess.run([sys.executable, 'scripts/generate.py', '--config', str(resolved)], check=True)
        generation_records.append({'name': key, 'heldout': cfg['output']})
    for key in ('v3a','v3b','v3c','v3d'):
        cfg = load_config('configs/generate_persona_' + key + '.yaml')
        resolved = Path('/kaggle/working/generate_eval_' + key + '.json'); write_json(resolved, cfg)
        subprocess.run([sys.executable, 'scripts/generate.py', '--config', str(resolved)], check=True)
        generation_records.append({'name': key, 'evaluation': cfg['output']})
    checks = {}
    for key, path in adapters.items():
        adapter_file = Path(path) / 'adapter_model.safetensors'
        checks[key] = {'sha256': file_hash(adapter_file), 'reload_verified':
                       read_json(Path(path).parent / 'reload_verified.json')['ok']}
    write_json('/kaggle/working/outputs/persona_v3_summary.json',
               {'environment': '/kaggle/working/outputs/environment.json', 'adapters': checks,
                'heldout_prompts': len(heldout_prompts), 'generations': generation_records,
                'sanity_preceded_full_generation': True})
    raise SystemExit(0)
if stage == 'dpo_hh_pilots':
    data = find_prepared_data('/kaggle/input', ('hh_manifest.json', 'hh_train.json', 'hh_validation.json'))
    markers = [p for p in Path('/kaggle/input').rglob('alignment_checkpoint.json')
               if read_json(p).get('track') == 'sft' and read_json(p).get('kind') == 'best']
    if len(markers) != 1:
        raise ValueError(f'Expected one attached SFT-best checkpoint, found {markers}')
    for name in ('dpo_hh_pilot_beta005', 'dpo_hh_pilot_beta01', 'dpo_hh_pilot_beta03'):
        pilot = load_config('configs/' + name + '.yaml')
        pilot['train_data'] = str(data / 'hh_train.json')
        pilot['validation_data'] = str(data / 'hh_validation.json')
        pilot['model'] = str(markers[0].parent)
        pilot['metrics_only'] = True
        resolved = Path('/kaggle/working/' + name + '_resolved.json')
        write_json(resolved, pilot)
        subprocess.run([sys.executable, 'scripts/train_dpo.py', '--config', str(resolved)], check=True)
    raise SystemExit(0)
config = load_config('configs/' + ('persona_lora' if stage == 'persona' else stage) + '.yaml')
if stage != 'persona':
    kind = 'sft' if stage == 'sft' else 'hh' if stage == 'dpo_hh' else 'ultrafeedback'
    data = find_prepared_data('/kaggle/input',
                              (kind + '_manifest.json', kind + '_train.json', kind + '_validation.json'))
    config['train_data'] = str(data / (kind + '_train.json'))
    config['validation_data'] = str(data / (kind + '_validation.json'))
    if stage != 'sft':
        markers = [p for p in Path('/kaggle/input').rglob('alignment_checkpoint.json')
                   if read_json(p).get('track') == 'sft' and read_json(p).get('kind') == 'best']
        if len(markers) != 1:
            raise ValueError(f'Expected one attached SFT-best checkpoint, found {markers}')
        config['model'] = str(markers[0].parent)
    script = 'train_sft' if stage == 'sft' else 'train_dpo'
else:
    script = 'train_persona'
resolved = Path('/kaggle/working/' + stage + '_run_config.yaml')
write_json(resolved, config)
print('STAGE', stage, 'TORCH', torch_version, 'GPUS', [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
subprocess.run([sys.executable, 'scripts/' + script + '.py', '--config', str(resolved)], check=True,
               env={**os.environ, **({'CUDA_VISIBLE_DEVICES': '0'} if stage == 'persona' else {})})
'''.replace("__STAGE_VALUE__", repr(stage)).replace("PAYLOAD", repr(payload))
    (out / "stage.py").write_text(template, encoding="utf-8")
    meta = {"id": "moibbi/" + STAGES[stage]["kernel"],
            "title": "PA2 " + stage.replace("_", " ").upper() + " Full",
            "code_file": "stage.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": STAGES[stage]["datasets"],
            "competition_sources": [], "kernel_sources": STAGES[stage]["kernels"]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {stage} stage: {out}. No job submitted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/prepare_kaggle_stage.py sft|dpo_hh|dpo_ultrafeedback|persona|persona_v2|persona_v3|dpo_hh_pilots")
    main(sys.argv[1])
