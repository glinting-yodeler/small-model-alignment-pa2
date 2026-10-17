"""Atomic optimizer-boundary state; load only trusted local training artifacts."""
import os
from pathlib import Path
import random
import torch


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_state(path, model, optimizer, scheduler, scaler, position, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(), "position": position, "identity": identity,
                "rng": rng_state()}, temp)
    os.replace(temp, path)


def load_state(path, model, optimizer, scheduler, scaler, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("Resume identity differs: config/data/model must match")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["position"]
