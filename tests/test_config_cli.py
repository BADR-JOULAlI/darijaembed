"""Check public configuration and command behavior without network access."""

import json
from pathlib import Path

import pytest

from darijaembed.cli import main
from darijaembed.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["local", "smoke", "semantic"])
def test_shipped_configs_load(name):
    config = load_config(ROOT / "configs" / f"{name}.toml")
    assert config.model.name
    assert config.dataset.role == "train"


def test_local_paths_follow_config_location():
    config = load_config(ROOT / "configs/local.toml")
    assert Path(config.dataset.path) == ROOT / "tests/fixtures/pairs.jsonl"
    assert Path(config.output_dir) == ROOT / "runs/local"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("validation_fraction = 0.25", "validation_fraction = 0.0", "validation_fraction"),
        ("batch_size = 2", "batch_size = 0", "batch_size"),
        ('precision = "no"', 'precision = "magic"', "precision"),
        ("seed = 42", "seed = true", "seed"),
    ],
)
def test_invalid_settings_rejected(tmp_path, old, new, message):
    text = (ROOT / "configs/local.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_prepare_cli_writes_auditable_splits(tmp_path, capsys):
    source = (ROOT / "tests/fixtures/pairs.jsonl").as_posix()
    path = tmp_path / "run.toml"
    path.write_text(
        f'''output_dir = "run"
[model]
name = "unused-no-download"
[dataset]
path = "{source}"
format = "jsonl"
[dataset.columns]
anchor = "anchor"
positive = "positive"
''',
        encoding="utf-8",
    )
    assert main(["prepare", "--config", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["input_rows"] == 8
    assert set(report["files_sha256"]) == {"train.jsonl", "validation.jsonl"}
    assert (tmp_path / "run/data/config.json").exists()


def test_cli_reports_bad_config_without_traceback(tmp_path, capsys):
    assert main(["prepare", "--config", str(tmp_path / "missing.toml")]) == 1
    assert "darijaembed:" in capsys.readouterr().err
