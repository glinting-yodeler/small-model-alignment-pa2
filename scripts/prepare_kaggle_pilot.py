"""Package exact local source in one CLI-uploadable kernel; never submit a job."""
import base64
import io
import json
from pathlib import Path
import hashlib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "kaggle_pilot"
    out.mkdir(exist_ok=True)
    buffer = io.BytesIO()
    roots = ["alignment", "scripts", "configs", "tests", "data/persona_lora", "eval_sets"]
    files = [root / f for f in ("pyproject.toml", "submission_adapter.py", "requirements-kaggle.txt")]
    for name in roots:
        files.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.append(root / "docs/starter_manifest.json")
    starter = json.loads((root / "docs/starter_manifest.json").read_text(encoding="utf-8"))
    files.extend(root / name for name in starter["files"])
    files = sorted(set(files))
    manifest = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root))
        z.writestr("source_manifest.json", json.dumps(manifest, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    bootstrap = '''import base64, io, os, subprocess, sys, zipfile, importlib.metadata
from pathlib import Path
root = Path('/kaggle/working/pa2-pilot-source')
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
os.chdir(root)
torch_version = importlib.metadata.version('torch')
Path('keep-kaggle-torch.txt').write_text('torch==' + torch_version + '\\n')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-kaggle.txt', '-c', 'keep-kaggle-torch.txt'], check=True)
assert importlib.metadata.version('torch') == torch_version
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'], check=True)
subprocess.run([sys.executable, '-m', 'pytest', '-q'],
               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}, check=True)
subprocess.run([sys.executable, 'scripts/run_pilot.py'], check=True)
'''.replace("PAYLOAD", repr(payload))
    (out / "pilot.py").write_text(bootstrap, encoding="utf-8")
    meta = json.loads((root / "kaggle_smoke/gpu/kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id=meta["id"].split("/")[0] + "/pa2-small-gpu-pilot", title="PA2 Small GPU Pilot",
                code_file="pilot.py", enable_gpu=True, enable_internet=True)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(files)} source files at {out}; no Kaggle job submitted.")


if __name__ == "__main__":
    main()
