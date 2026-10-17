"""Build separate reproducible generation kernels after model training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile


MODES = {
    "sft": {"kernel": "pa2-generate-sft", "source": "moibbi/pa2-sft-full",
            "dataset": "moibbi/pa2-frozen-prepared-inputs",
            "configs": ["generate_base_alpaca", "generate_sft_alpaca", "generate_sft_hh"]},
    "hh": {"kernel": "pa2-generate-hh", "source": "moibbi/pa2-dpo-hh-full",
           "dataset": None, "configs": ["generate_dpo_hh"]},
    "ultrafeedback": {"kernel": "pa2-generate-ultrafeedback", "source": "moibbi/pa2-dpo-ultrafeedback-full",
                      "dataset": "moibbi/pa2-frozen-prepared-inputs",
                      "configs": ["generate_ultrafeedback_alpaca"]},
    "persona": {"kernel": "pa2-generate-persona", "source": "moibbi/pa2-persona-full",
                "dataset": None, "configs": ["generate_persona_base", "generate_persona_lora"]},
    "persona_v2": {"kernel": "pa2-generate-persona-v2", "source": "moibbi/pa2-persona-v2-full",
                   "dataset": None, "configs": ["generate_persona_lora_v2", "generate_persona_lora_v2_lr1e4"]},
}


def main(mode):
    if mode not in MODES:
        raise ValueError(f"Choose one of {list(MODES)}")
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_generation" / mode
    allowed = {"generation.py", "kernel-metadata.json", "source_manifest.json", "__pycache__"}
    if out.exists() and any(p.name not in allowed for p in out.iterdir()):
        raise FileExistsError(f"Unexpected generated package contents: {out}")
    out.mkdir(parents=True, exist_ok=True)
    files = [p for directory in ("alignment", "configs", "eval_sets")
             for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [root / name for name in ("scripts/generate.py", "pyproject.toml", "requirements-kaggle.txt",
                                       "submission_adapter.py")]
    if mode == "persona_v2":
        files.append(root / "scripts/sanity_check_generation.py")
    files = sorted(set(files))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(root))
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    template = '''import base64, importlib.metadata, io, json, os, subprocess, sys, zipfile
from pathlib import Path
mode = __MODE_VALUE__
root = Path('/kaggle/working/pa2-generation-source')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(__PAYLOAD_VALUE__))).extractall(root)
os.chdir(root)
sys.path.insert(0, str(root))
torch_version = importlib.metadata.version('torch')
Path('keep-kaggle-torch.txt').write_text('torch==' + torch_version + '\\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-kaggle.txt',
                '-c', 'keep-kaggle-torch.txt'], check=True)
assert importlib.metadata.version('torch') == torch_version
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'], check=True)
import torch
assert torch.cuda.is_available()
from alignment.io import file_hash, load_config, read_json, write_json
from alignment.inputs import find_prepared_data
source_root = Path('/kaggle/input')
data = None
if mode in ('sft', 'ultrafeedback'):
    data = find_prepared_data(source_root, ('alpaca_eval_300_seed42.json',))

def unique_checkpoint(track):
    markers = [p for p in source_root.rglob('alignment_checkpoint.json')
               if (info := read_json(p)).get('track') == track and info.get('kind') == 'best']
    if len(markers) != 1:
        raise ValueError(f'Expected exactly one {track} best checkpoint, found {markers}')
    return markers[0].parent

checkpoints = {}
if mode not in ('persona', 'persona_v2'):
    track = 'sft' if mode == 'sft' else 'dpo_hh' if mode == 'hh' else 'dpo_ultrafeedback'
    checkpoints[track] = unique_checkpoint(track)
if mode in ('persona', 'persona_v2'):
    wanted = ('persona',) if mode == 'persona' else ('persona_v2', 'persona_v2_lr1e4')
    for run_name in wanted:
        adapters = [p.parent for p in source_root.rglob('adapter_config.json')
                    if p.parent.name == 'adapter' and p.parent.parent.name == run_name]
        if len(adapters) != 1:
            raise ValueError(f'Expected one final {run_name} adapter, found {adapters}')
        checkpoints[run_name] = adapters[0]

names = __CONFIG_NAMES__
results = {}
for name in names:
    config = load_config('configs/' + name + '.yaml')
    config['output'] = str(Path('/kaggle/working/outputs/generations') / Path(config['output']).name)
    if 'alpaca' in name:
        config['prompts'] = str(data / 'alpaca_eval_300_seed42.json')
    if name.startswith('generate_sft'):
        config['model'] = str(checkpoints['sft'])
    elif name == 'generate_dpo_hh':
        config['model'] = str(checkpoints['dpo_hh'])
    elif name == 'generate_ultrafeedback_alpaca':
        config['model'] = str(checkpoints['dpo_ultrafeedback'])
    elif name == 'generate_persona_lora':
        config['adapter'] = str(checkpoints['persona'])
    elif name == 'generate_persona_lora_v2':
        config['adapter'] = str(checkpoints['persona_v2'])
    elif name == 'generate_persona_lora_v2_lr1e4':
        config['adapter'] = str(checkpoints['persona_v2_lr1e4'])
    resolved = Path('/kaggle/working/' + name + '_resolved.json')
    write_json(resolved, config)
    if mode == 'persona_v2':
        sanity = Path('/kaggle/working/outputs/analysis') / ('sanity_' + Path(config['output']).stem + '.json')
        subprocess.run([sys.executable, 'scripts/sanity_check_generation.py', '--config', str(resolved),
                        '--count', '3', '--output', str(sanity)], check=True)
    subprocess.run([sys.executable, 'scripts/generate.py', '--config', str(resolved)], check=True)
    output = Path(config['output'])
    count = len(read_json(output))
    expected = 20 if mode in ('persona', 'persona_v2') else 300
    if count != expected:
        raise ValueError(f'{name}: expected {expected} generations, got {count}')
    results[name] = {'count': count, 'sha256': file_hash(output), 'path': str(output)}
write_json('/kaggle/working/outputs/generations/completion.json',
           {'mode': mode, 'checkpoints': {k: str(v) for k, v in checkpoints.items()},
            'results': results, 'torch': torch_version})
print('GENERATION_COMPLETE', mode, results, flush=True)
'''.replace("__MODE_VALUE__", repr(mode)).replace("__PAYLOAD_VALUE__", repr(payload))
    template = template.replace("__CONFIG_NAMES__", repr(MODES[mode]["configs"]))
    (out / "generation.py").write_text(template, encoding="utf-8")
    settings = MODES[mode]
    metadata = {"id": "moibbi/" + settings["kernel"], "title": "PA2 Generate " + mode,
                "code_file": "generation.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [settings["dataset"]] if settings["dataset"] else [],
                "competition_sources": [], "kernel_sources": [settings["source"]]}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {mode} generation package at {out}; no job submitted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/prepare_kaggle_generation.py sft|hh|ultrafeedback|persona|persona_v2")
    main(sys.argv[1])
