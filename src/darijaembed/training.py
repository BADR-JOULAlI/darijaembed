"""Contrastive fine-tuning with a held-out validation split."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .data import load_source, normalize_text, prepare_pairs, read_jsonl

if TYPE_CHECKING:
    from .config import ExperimentConfig


def resolve_device_and_precision(torch: Any, device: str, precision: str) -> tuple[str, str]:
    """Choose supported hardware without silently changing an explicit request."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("Training device must be 'auto', 'cpu', or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    if precision not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("Precision must be 'auto', 'no', 'fp16', or 'bf16'.")
    if precision == "auto":
        precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "no"
    if device == "cpu" and precision != "no":
        raise ValueError("This CPU training path requires precision = 'no' or 'auto'.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested, but the CUDA device does not support it.")
    return device, precision


def _validate_splits(train_rows: list[dict], validation_rows: list[dict]) -> list[str]:
    """Recheck prepared files instead of trusting a potentially stale audit."""
    if len(train_rows) < 2 or len(validation_rows) < 2:
        raise ValueError(
            "Contrastive training needs at least two training pairs and two validation pairs."
        )
    columns = ["anchor", "positive"]
    if "negative" in train_rows[0]:
        columns.append("negative")
    split_texts = []
    for label, rows in (("train", train_rows), ("validation", validation_rows)):
        texts = set()
        for index, row in enumerate(rows, start=1):
            if set(row) != set(columns):
                raise ValueError(f"{label} row {index} must have exactly these columns: {columns}.")
            values = []
            for column in columns:
                value = row[column]
                if not isinstance(value, str) or not normalize_text(value):
                    raise ValueError(f"{label} row {index}: {column} must contain nonempty text.")
                values.append(normalize_text(value))
            if len(set(values)) != len(values):
                raise ValueError(
                    f"{label} row {index} contains identical anchor, positive, or negative text."
                )
            texts.update(values)
        split_texts.append(texts)
    if split_texts[0] & split_texts[1]:
        raise ValueError("Train/validation leakage: normalized text occurs in both splits.")
    return columns


def _versions() -> dict[str, str]:
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "datasets", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": platform.python_version(), **packages}


def _split_fingerprint(rows: list[dict]) -> str:
    serialized = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _validate_output_dir(output_dir: Path) -> None:
    """Permit preprocessing and baseline results, but protect prior training runs."""
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"Output path must be a directory: {output_dir}.")
    unexpected = [
        path.name
        for path in output_dir.iterdir()
        if path.name not in {"data", "evaluation"} or not path.is_dir()
    ]
    if unexpected:
        raise ValueError(
            f"Output directory contains prior training artifacts or unexpected files: {output_dir} "
            f"({', '.join(sorted(unexpected))}). Choose a new run directory."
        )


def train(config: ExperimentConfig, *, prepared_dir: str | Path | None = None) -> dict:
    """Fine-tune one encoder and persist measured metrics and run provenance.

    Gradient accumulation changes optimizer update frequency. It does not pool
    negatives across microbatches in MultipleNegativesRankingLoss.
    """
    if config.dataset.role != "train":
        raise ValueError(
            "Training requires dataset.role = 'train'; held-out datasets cannot be used."
        )
    if config.training.batch_size < 2:
        raise ValueError("MultipleNegativesRankingLoss requires training.batch_size >= 2.")
    output_dir = Path(config.output_dir)
    _validate_output_dir(output_dir)
    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
        from transformers import set_seed
    except ImportError as error:
        raise RuntimeError("Training requires the ML extras: pip install -e '.[ml]'.") from error

    if prepared_dir is None:
        train_rows, validation_rows, audit = prepare_pairs(
            load_source(config.dataset), config.training.validation_fraction, config.seed
        )
    else:
        prepared_dir = Path(prepared_dir)
        train_rows = read_jsonl(prepared_dir / "train.jsonl")
        validation_rows = read_jsonl(prepared_dir / "validation.jsonl")
        audit = json.loads((prepared_dir / "audit.json").read_text(encoding="utf-8"))
    columns = _validate_splits(train_rows, validation_rows)
    # Column insertion order tells Sentence Transformers which text is the query.
    train_data = Dataset.from_list([{key: row[key] for key in columns} for row in train_rows])
    validation_data = Dataset.from_list(
        [{key: row[key] for key in columns} for row in validation_rows]
    )
    device, precision = resolve_device_and_precision(
        torch, config.model.device, config.training.precision
    )
    set_seed(config.seed)
    model = SentenceTransformer(config.model.name, revision=config.model.revision, device=device)
    model.max_seq_length = config.model.max_seq_length
    batch_size = min(config.training.batch_size, len(train_rows))
    prompts = None
    if config.model.query_prompt or config.model.document_prompt:
        prompts = {
            key: config.model.query_prompt if key == "anchor" else config.model.document_prompt
            for key in columns
        }
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        use_cpu=device == "cpu",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda",
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        save_strategy="no",
        report_to="none",
        prompts=prompts,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": asdict(config),
        "versions": _versions(),
        "runtime": {"device": device, "precision": precision, "batch_size": batch_size},
        "audit": audit,
        "splits": {
            "train": {"rows": len(train_rows), "sha256": _split_fingerprint(train_rows)},
            "validation": {
                "rows": len(validation_rows),
                "sha256": _split_fingerprint(validation_rows),
            },
        },
        "prepared_dir": str(prepared_dir) if prepared_dir is not None else None,
        "status": "started",
    }
    run_file = output_dir / "run.json"
    run_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    training_result = trainer.train()
    result["train_metrics"] = training_result.metrics
    result["validation_metrics"] = trainer.evaluate()
    model.save_pretrained(str(output_dir / "model"))
    result["status"] = "completed"
    run_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
