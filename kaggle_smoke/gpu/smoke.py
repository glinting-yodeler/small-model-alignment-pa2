import json
import platform
import subprocess
from pathlib import Path
import torch

subprocess.run(["nvidia-smi"], check=True)

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 2, (
    f"Expected 2 GPUs, found {torch.cuda.device_count()}"
)

devices = []
for i in range(2):
    name = torch.cuda.get_device_name(i)
    assert "T4" in name, f"Expected T4, found {name}"

    device = torch.device(f"cuda:{i}")
    a = torch.ones((128, 128), device=device)
    result = a @ a
    torch.cuda.synchronize(device)
    assert torch.allclose(result, torch.full_like(result, 128))

    devices.append({
        "index": i,
        "name": name,
        "memory_gib": round(
            torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
        ),
        "computation_passed": True
    })

report = {
    "ok": True,
    "stage": "gpu",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "devices": devices
}

Path("/kaggle/working/gpu_smoke_result.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
