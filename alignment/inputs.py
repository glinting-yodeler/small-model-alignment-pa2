"""Locate versioned Kaggle inputs across supported mount layouts."""
from pathlib import Path
from .io import file_hash, read_json


def find_prepared_data(input_root, required):
    root = Path(input_root)
    markers = sorted(root.rglob("upload_file_hashes.json"))
    if len(markers) != 1:
        entries = sorted(str(p) for p in root.glob("*"))
        raise ValueError(f"Expected one frozen prepared dataset below {root}; found {markers}; entries={entries}")
    data = markers[0].parent
    hashes = read_json(markers[0])
    for name in ("manifest.json", *required):
        if name not in hashes or file_hash(data / name) != hashes[name]:
            raise ValueError(f"Missing or mismatched prepared input: {data / name}")
    return data
