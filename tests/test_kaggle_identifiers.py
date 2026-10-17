"""Future DPO training kernels must match generation's attached source."""
import json

import pytest

from scripts import prepare_kaggle_generation, prepare_kaggle_stage


@pytest.mark.parametrize("stage,mode", [
    ("dpo_hh", "hh"),
    ("dpo_ultrafeedback", "ultrafeedback"),
])
def test_dpo_stage_kernel_id_matches_generation_source(tmp_path, monkeypatch, stage, mode):
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("pyproject.toml", "requirements-kaggle.txt", "submission_adapter.py"):
        (root / name).write_text("", encoding="utf-8")
    (scripts / "generate.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(prepare_kaggle_stage, "__file__", str(scripts / "prepare_kaggle_stage.py"))
    monkeypatch.setattr(prepare_kaggle_generation, "__file__", str(scripts / "prepare_kaggle_generation.py"))

    prepare_kaggle_stage.main(stage)
    prepare_kaggle_generation.main(mode)
    training = json.loads((root / "kaggle_stages" / stage / "kernel-metadata.json").read_text(encoding="utf-8"))
    generation = json.loads((root / "kaggle_generation" / mode / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert generation["kernel_sources"] == [training["id"]]
