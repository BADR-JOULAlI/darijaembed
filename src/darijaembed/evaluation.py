"""Evaluate sentence embeddings against a held-out similarity dataset."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def compute_metrics(
    scores: Sequence[float],
    similarities: Sequence[float],
    *,
    score_min: float,
    score_max: float,
) -> dict[str, float | int]:
    """Compute STS correlations, rejecting undefined or malformed inputs."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    labels = _validate_scores(scores, score_min=score_min, score_max=score_max)
    predictions = np.asarray(similarities, dtype=float)
    if labels.ndim != 1 or predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("Scores and similarities must be one-dimensional and equally sized.")
    if not np.isfinite(predictions).all():
        raise ValueError("Scores and similarities must contain only finite numbers.")
    if np.ptp(predictions) == 0:
        raise ValueError("Correlation is undefined for constant scores or similarities.")
    return {
        "pair_count": int(len(labels)),
        "spearman": float(spearmanr(labels, predictions).statistic),
        "pearson": float(pearsonr(labels, predictions).statistic),
        "mean_cosine_similarity": float(predictions.mean()),
    }


def _validate_scores(scores: Sequence[float], *, score_min: float, score_max: float):
    import numpy as np

    if not np.isfinite([score_min, score_max]).all() or score_min >= score_max:
        raise ValueError("score_min must be finite and smaller than score_max.")
    labels = np.asarray(scores, dtype=float)
    if labels.ndim != 1:
        raise ValueError("Scores must be one-dimensional.")
    if len(labels) < 3:
        raise ValueError("STS evaluation requires at least three scored sentence pairs.")
    if not np.isfinite(labels).all():
        raise ValueError("Scores must contain only finite numbers.")
    if np.any(labels < score_min) or np.any(labels > score_max):
        raise ValueError(f"Scores must lie within [{score_min}, {score_max}].")
    if np.ptp(labels) == 0:
        raise ValueError("Correlation is undefined for constant scores or similarities.")
    return labels


def _read_pairs(rows: list[dict]) -> tuple[list[str], list[str], list[float]]:
    left, right, scores = [], [], []
    for index, row in enumerate(rows):
        for column in ("sentence1", "sentence2"):
            text = row.get(column)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Evaluation row {index}: {column} must be nonempty text.")
        raw_score = row.get("score")
        if isinstance(raw_score, bool):
            raise ValueError(f"Evaluation row {index}: score must be numeric, not boolean.")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Evaluation row {index}: score must be numeric.") from error
        left.append(row["sentence1"])
        right.append(row["sentence2"])
        scores.append(score)
    return left, right, scores


def evaluate(config: ExperimentConfig, *, model_name: str | None = None) -> dict:
    """Encode held-out pairs and save a model-specific, auditable STS report."""
    import numpy as np

    from .data import load_source

    settings = config.evaluation
    if settings is None:
        raise ValueError("An [evaluation] configuration is required.")
    rows = load_source(settings.dataset)
    left, right, scores = _read_pairs(rows)
    # Validate labels before loading a model or downloading its weights.
    _validate_scores(
        scores,
        score_min=settings.score_min,
        score_max=settings.score_max,
    )

    from sentence_transformers import SentenceTransformer

    selected_model = model_name or config.model.name
    revision = config.model.revision if model_name is None else None
    device = None if config.model.device == "auto" else config.model.device
    model = SentenceTransformer(selected_model, revision=revision, device=device)
    model.max_seq_length = config.model.max_seq_length
    encode_options = {
        "batch_size": settings.batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    embeddings_left = model.encode(left, prompt=config.model.query_prompt, **encode_options)
    embeddings_right = model.encode(right, prompt=config.model.document_prompt, **encode_options)
    similarities = np.clip(np.sum(embeddings_left * embeddings_right, axis=1), -1.0, 1.0)
    metrics = compute_metrics(
        scores,
        similarities,
        score_min=settings.score_min,
        score_max=settings.score_max,
    )

    versions = {}
    for package in ("darijaembed", "sentence-transformers", "torch", "datasets", "numpy", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    fingerprint = sha256(
        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    model_key = sha256(f"{selected_model}@{revision}".encode()).hexdigest()[:8]
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", selected_model).strip("-.")[-80:] or "model"
    destination = Path(config.output_dir) / "evaluation" / f"{safe_name}-{model_key}.json"
    report = {
        "model": selected_model,
        "model_revision": revision,
        "device": str(model.device),
        "dataset_sha256": fingerprint,
        "metrics": metrics,
        "config": asdict(config),
        "versions": versions,
        "report_path": str(destination),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
