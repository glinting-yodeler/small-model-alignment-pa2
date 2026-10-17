import json
import platform
from pathlib import Path

result = {
    "ok": True,
    "stage": "cpu",
    "python": platform.python_version(),
    "checksum": sum(range(100))
}

path = Path("/kaggle/working/cpu_smoke_result.json")
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
