"""Load configured datasets and prepare sentence-disjoint contrastive splits."""

from __future__ import annotations

import csv
import json
import random
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DatasetConfig


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without changing Darija spelling."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON objects, reporting the location of malformed rows."""
    return list(_iter_jsonl(Path(path)))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write portable, readable records; never escape Arabic characters."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_source(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load local JSONL/CSV or a Hugging Face split and map configured columns.

    The row limit is applied before converting the source to Python records.
    Hugging Face authentication uses its standard local login or HF_TOKEN.
    """
    if config.max_samples is not None and config.max_samples < 1:
        raise ValueError("dataset.max_samples must be positive")
    if not config.columns:
        raise ValueError("dataset.columns must map canonical names to source columns")

    path = Path(config.path)
    source_format = config.format
    if source_format is None:
        suffix = path.suffix.lower().lstrip(".")
        source_format = suffix if suffix in {"jsonl", "csv"} or path.is_file() else "huggingface"
    if source_format == "jsonl":
        rows = islice(_iter_jsonl(path), config.max_samples)
        return _map_columns(rows, config.columns)
    if source_format == "csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = islice(csv.DictReader(handle), config.max_samples)
            return _map_columns(rows, config.columns)
    if source_format not in {"huggingface", "hf"}:
        raise ValueError(
            f"Unsupported dataset format {source_format!r}; use jsonl, csv or huggingface"
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face loading requires the training dependencies") from exc

    dataset = load_dataset(
        config.path,
        name=config.name,
        split=config.split,
        revision=config.revision,
    )
    if config.max_samples is not None:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    return _map_columns(dataset, config.columns)


def _map_columns(rows: Iterable[dict[str, Any]], columns: dict[str, str]) -> list[dict[str, Any]]:
    mapped = []
    for number, row in enumerate(rows, start=1):
        missing = sorted(set(columns.values()) - row.keys())
        if missing:
            raise ValueError(f"Dataset row {number} is missing configured columns: {missing}")
        mapped.append({canonical: row[source] for canonical, source in columns.items()})
    if not mapped:
        raise ValueError("The configured dataset contains no rows")
    return mapped


def prepare_pairs(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Clean pairs/triplets and split entire connected sentence components.

    If any sentence occurs in two records, those records remain in the same
    split, including transitive links through hard negatives. Duplicate positive
    pairs are removed irrespective of orientation; the first occurrence wins.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    audit: dict[str, Any] = {
        "input_rows": len(rows),
        "valid_rows": 0,
        "invalid_rows": 0,
        "self_pairs": 0,
        "duplicate_rows": 0,
    }
    for row in rows:
        fields = ("anchor", "positive", "negative") if "negative" in row else ("anchor", "positive")
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in fields):
            audit["invalid_rows"] += 1
            continue
        pair = {field: normalize_text(row[field]) for field in fields}
        if pair["anchor"] == pair["positive"]:
            audit["self_pairs"] += 1
            continue
        if pair.get("negative") in (pair["anchor"], pair["positive"]):
            audit["invalid_rows"] += 1
            continue
        key = tuple(sorted((pair["anchor"], pair["positive"])))
        if key in seen:
            audit["duplicate_rows"] += 1
            continue
        seen.add(key)
        cleaned.append(pair)

    audit["valid_rows"] = len(cleaned)
    if len({frozenset(row) for row in cleaned}) > 1:
        raise ValueError(
            "Use either pairs or triplets in one dataset; negative columns cannot be mixed"
        )
    components = _sentence_components(cleaned)
    if len(components) < 2:
        raise ValueError(
            "Need at least two disconnected sentence groups for a leakage-free split; "
            "provide more independent pairs or audit repeated sentences"
        )

    random.Random(seed).shuffle(components)
    target = len(cleaned) * validation_fraction
    validation_indices: set[int] = set()
    selected_components = 0
    for indices in components:
        current_distance = abs(len(validation_indices) - target)
        candidate_distance = abs(len(validation_indices) + len(indices) - target)
        if candidate_distance < current_distance and selected_components < len(components) - 1:
            validation_indices.update(indices)
            selected_components += 1
    if not validation_indices:
        validation_indices.update(min(components, key=len))

    train = [row for index, row in enumerate(cleaned) if index not in validation_indices]
    validation = [row for index, row in enumerate(cleaned) if index in validation_indices]
    audit.update(
        components=len(components),
        train_rows=len(train),
        validation_rows=len(validation),
        validation_fraction_actual=len(validation) / len(cleaned),
    )
    return train, validation, audit


def _sentence_components(rows: list[dict[str, str]]) -> list[list[int]]:
    """Group record indices with a small union-find over their sentence texts."""
    parent: dict[str, str] = {}

    def find(text: str) -> str:
        parent.setdefault(text, text)
        root = text
        while parent[root] != root:
            root = parent[root]
        while parent[text] != text:
            next_text = parent[text]
            parent[text] = root
            text = next_text
        return root

    for row in rows:
        anchor_root = find(row["anchor"])
        for text in row.values():
            parent[find(text)] = anchor_root

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(row["anchor"])].append(index)
    return list(grouped.values())
