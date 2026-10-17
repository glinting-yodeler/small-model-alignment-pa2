"""Small real GPU gate, independent of the cancelled standalone smoke job."""
import copy
import gc
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.io import load_config, write_json, read_json, environment


def main():
    import torch
    from alignment.preparation import prepare
    from alignment.training import run_training
    from alignment.generation import generate
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    assert all("T4" in torch.cuda.get_device_name(i) for i in range(2))
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        a = torch.ones((128, 128), device=device)
        result = a @ a
        torch.cuda.synchronize(device)
        assert torch.allclose(result, torch.full_like(result, 128))
    out = Path("/kaggle/working/pa2-pilot")
    if out.exists():
        raise FileExistsError("Pilot output already exists; inspect artifacts before rerunning")
    started = time.monotonic()
    prep = load_config("configs/data.yaml")
    prep["output"] = str(out / "data")
    prepare(prep, pilot=True)
    for track in ("sft", "dpo_hh", "dpo_ultrafeedback"):
        cfg = load_config(f"configs/{track}.yaml")
        key = "sft" if track == "sft" else "hh" if track == "dpo_hh" else "ultrafeedback"
        cfg.update(output=str(out / track), train_data=str(out / "data" / key / "train.json"),
                   validation_data=str(out / "data" / key / "validation.json"),
                   accumulation=2, max_updates=2, eval_every=1, save_every=1, pilot_audit=True,
                   required_train_count=16, required_validation_count=4)
        if track != "sft":
            cfg["model"] = str(out / "sft/best")
        run_training(cfg)
        # Reload inference weights, then resume completed state (no extra updates).
        run_training(cfg, resume=str(out / track / "latest.pt"))
        # Keep best plus audits; discard verified pilot-only redundant weights to
        # avoid three AdamW states and duplicate models exhausting Kaggle disk.
        import shutil
        for path in [out / track / "latest.pt", out / track / "final"]:
            resolved = path.resolve()
            if not resolved.is_relative_to(out.resolve()):
                raise ValueError("Pilot cleanup escaped its output directory")
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        gc.collect()
        torch.cuda.empty_cache()
    prompts = read_json(out / "data/alpaca_eval_300_seed42.json")[:3]
    write_json(out / "prompts.json", prompts)
    gen = load_config("configs/generation.yaml")
    gen.update(model=str(out / "sft/best"), prompts=str(out / "prompts.json"),
               output=str(out / "sft_generations.json"), generator="pilot-sft", max_new_tokens=48)
    generate(gen)
    gc.collect()
    torch.cuda.empty_cache()
    persona = load_config("configs/persona_lora.yaml")
    persona.update(output=str(out / "persona"), max_updates=1, accumulation=1)
    # One policy GPU for persona, with an isolated TRL runtime.
    import os
    import subprocess
    write_json(out / "persona_config.json", persona)
    subprocess.run([sys.executable, "scripts/train_persona.py", "--config", str(out / "persona_config.json")],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, check=True)
    gc.collect()
    torch.cuda.empty_cache()
    gen.update(model=persona["model"], revision=persona["revision"], adapter=str(out / "persona/adapter"), format="chat",
               generator="pilot-persona", output=str(out / "persona_generations.json"))
    generate(gen)
    report = {**environment(), "ok": True, "pilot_only": True, "seconds": time.monotonic() - started,
              "devices": [torch.cuda.get_device_name(i) for i in range(2)],
              "peak_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
              "disk_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    write_json(out / "pilot_result.json", report)
    print(report)


if __name__ == "__main__":
    main()
