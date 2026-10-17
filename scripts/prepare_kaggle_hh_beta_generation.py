"""Prepare a single Kaggle job that retrains the matched HH beta pilots and generates immediately."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_stages" / "hh_beta_generation"
    out.mkdir(parents=True, exist_ok=True)
    files = [p for directory in ("alignment", "configs", "eval_sets")
             for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [root / name for name in (
        "scripts/train_dpo.py", "scripts/generate.py", "scripts/sanity_check_generation.py",
        "pyproject.toml", "requirements-kaggle.txt", "submission_adapter.py")]
    files = sorted(set(files))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    runner = r'''import base64, importlib.metadata, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
root = Path('/kaggle/working/pa2-beta-generation-source')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(__PAYLOAD__))).extractall(root)
os.chdir(root); sys.path.insert(0, str(root))
torch_version = importlib.metadata.version('torch')
Path('keep-kaggle-torch.txt').write_text('torch==' + torch_version + '\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-kaggle.txt',
                '-c', 'keep-kaggle-torch.txt'], check=True)
assert importlib.metadata.version('torch') == torch_version
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'], check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
from alignment.io import file_hash, load_config, read_json, write_json
from alignment.inputs import find_prepared_data
source = Path('/kaggle/input')
data = find_prepared_data(source, ('hh_manifest.json', 'hh_train.json', 'hh_validation.json'))
markers = [p for p in source.rglob('alignment_checkpoint.json')
           if read_json(p).get('track') == 'sft' and read_json(p).get('kind') == 'best']
if len(markers) != 1: raise ValueError(f'Expected one SFT best checkpoint, found {markers}')
sft = markers[0].parent
output_root = Path('/kaggle/working/outputs')
summary = {'torch': torch_version, 'gpus': [torch.cuda.get_device_name(i) for i in range(2)], 'runs': []}
for name, label in [('dpo_hh_pilot_beta005','beta005'), ('dpo_hh_pilot_beta01','beta01'),
                    ('dpo_hh_pilot_beta03','beta03')]:
    train = load_config('configs/' + name + '.yaml')
    train.update(model=str(sft), train_data=str(data/'hh_train.json'),
                 validation_data=str(data/'hh_validation.json'),
                 output=str(output_root/'training'/label), metrics_only=False,
                 skip_resume_state=True, save_every=1000000)
    train_path = Path('/kaggle/working/' + label + '_train.json'); write_json(train_path, train)
    subprocess.run([sys.executable, 'scripts/train_dpo.py', '--config', str(train_path)], check=True)
    checkpoint = Path(train['output'])/'best'
    generation = {'model': str(checkpoint),
                  'prompts': str(root/'eval_sets/hh_rlhf_eval_300_seed42.json'),
                  'output': str(output_root/'generations'/(label + '.json')),
                  'generator': 'hh-' + label + '-150', 'device': 'cuda:0', 'batch_size': 2,
                  'max_new_tokens': 256, 'repetition_penalty': 1.0, 'format': 'plain'}
    generation_path = Path('/kaggle/working/' + label + '_generation.json')
    write_json(generation_path, generation)
    sanity = output_root/'sanity'/(label + '.json')
    subprocess.run([sys.executable, 'scripts/sanity_check_generation.py', '--config', str(generation_path),
                    '--count', '3', '--output', str(sanity)], check=True)
    subprocess.run([sys.executable, 'scripts/generate.py', '--config', str(generation_path)], check=True)
    rows = read_json(generation['output'])
    if len(rows) != 300 or len({row['id'] for row in rows}) != 300:
        raise ValueError(f'{label}: invalid generation coverage')
    training_dir = Path(train['output'])
    for directory in ('best', 'final'):
        shutil.rmtree(training_dir/directory, ignore_errors=True)
    summary['runs'].append({'label': label, 'beta': train['beta'], 'updates': 150,
                            'generation_count': len(rows),
                            'generation_sha256': file_hash(generation['output']),
                            'sanity_sha256': file_hash(sanity)})
write_json(output_root/'beta_generation_summary.json', summary)
print('BETA_GENERATION_COMPLETE', json.dumps(summary), flush=True)
'''.replace('__PAYLOAD__', repr(payload))
    (out / "run.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": "moibbi/pa2-hh-beta-generation", "title": "PA2 HH Beta Generation",
        "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["moibbi/pa2-frozen-prepared-inputs"],
        "competition_sources": [], "kernel_sources": ["moibbi/pa2-sft-full"]}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {out}; no job submitted.")


if __name__ == "__main__":
    main()
