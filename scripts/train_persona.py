"""CLI for train_persona."""
from pathlib import Path
import sys
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.cli import main

if __name__ == "__main__":
    main('train_persona')
