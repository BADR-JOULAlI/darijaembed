"""Check training wiring without downloading a model or importing PyTorch."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from darijaembed import training
from darijaembed.config import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from darijaembed.data import write_jsonl


def fake_torch(*, cuda=False, bf16=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16)
    )


def test_explicit_cpu_does_not_enable_gpu_precision():
    assert training.resolve_device_and_precision(
        fake_torch(cuda=True, bf16=True), "cpu", "auto"
    ) == ("cpu", "no")


def test_explicit_unavailable_hardware_or_precision_fails():
    with pytest.raises(ValueError, match="cannot access"):
        training.resolve_device_and_precision(fake_torch(), "cuda", "auto")
    with pytest.raises(ValueError, match="does not support"):
        training.resolve_device_and_precision(fake_torch(cuda=True), "cuda", "bf16")
    with pytest.raises(ValueError, match="CPU"):
        training.resolve_device_and_precision(fake_torch(), "cpu", "fp16")


def test_prepared_split_leakage_including_negatives_is_rejected():
    train_rows = [
        {"anchor": "a", "positive": "b", "negative": "c"},
        {"anchor": "d", "positive": "e", "negative": "f"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        training._validate_splits(
            train_rows,
            [
                {"anchor": "g", "positive": "h", "negative": " c "},
                {"anchor": "j", "positive": "k", "negative": "l"},
            ],
        )


def test_prepared_mixed_schema_and_negative_collisions_are_rejected():
    rows = [{"anchor": "a", "positive": "b"}, {"anchor": "c", "positive": "d"}]
    with pytest.raises(ValueError, match="columns"):
        training._validate_splits(
            rows,
            [
                {"anchor": "e", "positive": "f", "negative": "g"},
                {"anchor": "h", "positive": "i", "negative": "j"},
            ],
        )
    rows = [
        {"anchor": "a", "positive": "b", "negative": "a"},
        {"anchor": "d", "positive": "e", "negative": "f"},
    ]
    with pytest.raises(ValueError, match="identical"):
        training._validate_splits(
            rows,
            [
                {"anchor": "g", "positive": "h", "negative": "i"},
                {"anchor": "j", "positive": "k", "negative": "l"},
            ],
        )


def test_contrastive_validation_needs_at_least_two_rows():
    with pytest.raises(ValueError, match="two validation"):
        training._validate_splits(
            [{"anchor": "a", "positive": "b"}, {"anchor": "c", "positive": "d"}],
            [{"anchor": "e", "positive": "f"}],
        )


@pytest.mark.parametrize("artifact", ["run.json", "model", "checkpoint-1", "unexpected.txt"])
def test_output_directory_rejects_prior_runs_and_unexpected_files(tmp_path, artifact):
    output_dir = tmp_path / "run"
    (output_dir / "data").mkdir(parents=True)
    (output_dir / artifact).touch()
    with pytest.raises(ValueError, match="prior training artifacts"):
        training._validate_output_dir(output_dir)


def experiment_config(output_dir):
    return ExperimentConfig(
        model=ModelConfig(
            name="local/test-model",
            revision="fixed-revision",
            max_seq_length=32,
            device="cpu",
            query_prompt="query: ",
            document_prompt="passage: ",
        ),
        dataset=DatasetConfig(path="unused", columns={"anchor": "anchor", "positive": "positive"}),
        output_dir=str(output_dir),
        seed=23,
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=1,
            warmup_steps=0,
            eval_steps=1,
            logging_steps=1,
            gradient_checkpointing=False,
        ),
    )


def test_held_out_dataset_cannot_be_trained(tmp_path):
    config = experiment_config(tmp_path / "run")
    config = replace(config, dataset=replace(config.dataset, role="test"))
    with pytest.raises(ValueError, match="held-out"):
        training.train(config)


def test_batch_size_one_is_rejected_before_importing_ml_dependencies(tmp_path):
    config = experiment_config(tmp_path / "run")
    config = replace(config, training=replace(config.training, batch_size=1))
    with pytest.raises(ValueError, match="batch_size >= 2"):
        training.train(config)


def test_train_preserves_hard_negatives_prompts_and_measured_metrics(tmp_path, monkeypatch):
    calls = {}

    class FakeModel:
        def __init__(self, name, **kwargs):
            calls["model"] = {"name": name, **kwargs}

        def save_pretrained(self, destination):
            calls["saved_model"] = destination

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls["trainer"] = kwargs

        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.6})

        def evaluate(self):
            return {"eval_loss": 0.7}

    modules = {
        "torch": fake_torch(cuda=True, bf16=True),
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        "sentence_transformers": SimpleNamespace(
            SentenceTransformer=FakeModel,
            SentenceTransformerTrainer=FakeTrainer,
            SentenceTransformerTrainingArguments=lambda **kwargs: kwargs,
            losses=SimpleNamespace(
                MultipleNegativesRankingLoss=lambda model: ("ranking-loss", model)
            ),
        ),
        "sentence_transformers.training_args": SimpleNamespace(
            BatchSamplers=SimpleNamespace(NO_DUPLICATES="no-duplicates")
        ),
        "transformers": SimpleNamespace(set_seed=lambda seed: calls.update(seed=seed)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(training, "_versions", lambda: {"python": "test"})
    output_dir = tmp_path / "run"
    prepared = output_dir / "data"
    train_rows = [
        {"anchor": "a", "positive": "b", "negative": "c"},
        {"anchor": "d", "positive": "e", "negative": "f"},
    ]
    write_jsonl(prepared / "train.jsonl", train_rows)
    write_jsonl(
        prepared / "validation.jsonl",
        [
            {"anchor": "g", "positive": "h", "negative": "i"},
            {"anchor": "j", "positive": "k", "negative": "l"},
        ],
    )
    (prepared / "audit.json").write_text('{"train_rows": 2}', encoding="utf-8")
    (output_dir / "evaluation").mkdir()
    (output_dir / "evaluation" / "baseline.json").write_text("{}", encoding="utf-8")
    config = experiment_config(output_dir)
    result = training.train(config, prepared_dir=prepared)

    assert calls["trainer"]["train_dataset"] == train_rows
    assert list(calls["trainer"]["train_dataset"][0]) == ["anchor", "positive", "negative"]
    arguments = calls["trainer"]["args"]
    assert arguments["use_cpu"] is True
    assert arguments["bf16"] is False
    assert arguments["batch_sampler"] == "no-duplicates"
    assert arguments["prompts"] == {
        "anchor": "query: ",
        "positive": "passage: ",
        "negative": "passage: ",
    }
    assert arguments["gradient_accumulation_steps"] == 2
    assert result["train_metrics"] == {"train_loss": 0.6}
    assert result["validation_metrics"] == {"eval_loss": 0.7}
    assert result["status"] == "completed"
    assert len(result["splits"]["train"]["sha256"]) == 64
    saved = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert saved == result
    with pytest.raises(ValueError, match="prior training artifacts"):
        training.train(config, prepared_dir=prepared)
