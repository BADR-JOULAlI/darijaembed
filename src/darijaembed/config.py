"""Typed experiment settings loaded from TOML, without model-specific branches."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str | None = None
    max_seq_length: int = 128
    device: str = "auto"
    query_prompt: str = ""
    document_prompt: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    columns: dict[str, str]
    split: str = "train"
    name: str | None = None
    revision: str | None = None
    format: str | None = None
    max_samples: int | None = None
    role: str = "train"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    max_steps: int = 100
    warmup_steps: int = 10
    precision: str = "auto"
    validation_fraction: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 25
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: DatasetConfig
    batch_size: int = 32
    score_min: float = 0.0
    score_max: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    output_dir: str
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig | None = None


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _dataset(settings: dict[str, Any], base: Path) -> DatasetConfig:
    dataset = DatasetConfig(**settings)
    if not isinstance(dataset.path, str) or not dataset.path.strip():
        raise ValueError("dataset.path must be a nonempty string.")
    if not dataset.columns or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in dataset.columns.items()
    ):
        raise ValueError("dataset.columns must map canonical names to source column names.")
    if dataset.max_samples is not None:
        _positive_int(dataset.max_samples, "dataset.max_samples")
    if dataset.role not in {"train", "validation", "test"}:
        raise ValueError("dataset.role must be train, validation or test.")
    if dataset.format not in {None, "jsonl", "csv", "huggingface"}:
        raise ValueError("dataset.format must be jsonl, csv or huggingface.")
    local = dataset.format in {"jsonl", "csv"} or (
        dataset.format is None and Path(dataset.path).suffix in {".jsonl", ".csv"}
    )
    if local:
        settings = {**settings, "path": str((base / dataset.path).resolve())}
        dataset = DatasetConfig(**settings)
    return dataset


def load_config(path: str | Path) -> ExperimentConfig:
    """Resolve local paths relative to the config file, then validate settings."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = raw.keys() - {"model", "dataset", "output_dir", "seed", "training", "evaluation"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    model_settings = dict(raw["model"])
    local_model = path.parent / model_settings["name"]
    if local_model.is_dir():
        model_settings["name"] = str(local_model.resolve())
    model = ModelConfig(**model_settings)
    dataset = _dataset(raw["dataset"], path.parent)
    training = TrainingConfig(**raw.get("training", {}))
    evaluation = None
    if "evaluation" in raw:
        settings = dict(raw["evaluation"])
        evaluation = EvaluationConfig(
            dataset=_dataset(settings.pop("dataset"), path.parent), **settings
        )
        _positive_int(evaluation.batch_size, "evaluation.batch_size")
        if not evaluation.score_min < evaluation.score_max:
            raise ValueError("evaluation.score_min must be smaller than score_max.")
        required = {"sentence1", "sentence2", "score"}
        if set(evaluation.dataset.columns) != required:
            raise ValueError(f"Evaluation columns must be {sorted(required)}.")
    if not isinstance(model.name, str) or not model.name.strip():
        raise ValueError("model.name must be a nonempty string.")
    _positive_int(model.max_seq_length, "model.max_seq_length")
    if model.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("model.device must be auto, cpu or cuda.")
    for name in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "logging_steps",
        "eval_steps",
    ):
        _positive_int(getattr(training, name), f"training.{name}")
    if type(training.warmup_steps) is not int or training.warmup_steps < 0:
        raise ValueError("training.warmup_steps must be a nonnegative integer.")
    if not 0 < training.validation_fraction < 1:
        raise ValueError("training.validation_fraction must be between 0 and 1.")
    if not 0 < training.learning_rate < 1:
        raise ValueError("training.learning_rate must be between 0 and 1.")
    if training.precision not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("training.precision must be auto, no, fp16 or bf16.")
    if type(training.gradient_checkpointing) is not bool:
        raise ValueError("training.gradient_checkpointing must be a boolean.")
    seed = raw.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not {"anchor", "positive"} <= dataset.columns.keys():
        raise ValueError("Training dataset needs anchor and positive columns.")
    if dataset.columns.keys() - {"anchor", "positive", "negative"}:
        raise ValueError("Training columns may only contain anchor, positive and negative.")
    output_dir = raw["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a nonempty string.")
    return ExperimentConfig(
        model=model,
        dataset=dataset,
        output_dir=str((path.parent / output_dir).resolve()),
        seed=seed,
        training=training,
        evaluation=evaluation,
    )
