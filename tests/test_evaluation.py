"""STS metric and evaluation behavior without downloading models."""

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from darijaembed.evaluation import compute_metrics, evaluate


def test_metrics_preserve_order_and_handle_ties():
    metrics = compute_metrics([0, 2, 2, 5], [-0.5, 0.2, 0.2, 1], score_min=0, score_max=5)
    assert metrics["pair_count"] == 4
    assert metrics["spearman"] == pytest.approx(1)
    assert 0.99 < metrics["pearson"] <= 1


@pytest.mark.parametrize(
    ("scores", "predictions", "message"),
    [
        ([0, 1], [0.1, 0.9], "at least three"),
        ([0, 1, 2], [0.1, 0.9], "equally sized"),
        ([0, 1, float("nan")], [0.1, 0.5, 0.9], "finite"),
        ([0, 1, 2], [0.1, float("inf"), 0.9], "finite"),
        ([0, 1, 6], [0.1, 0.5, 0.9], "within"),
        ([1, 1, 1], [0.1, 0.5, 0.9], "constant"),
        ([0, 1, 2], [0.2, 0.2, 0.2], "constant"),
        ([[0], [1], [2]], [0.1, 0.5, 0.9], "one-dimensional"),
    ],
)
def test_invalid_metrics_fail_clearly(scores, predictions, message):
    with pytest.raises(ValueError, match=message):
        compute_metrics(scores, predictions, score_min=0, score_max=5)


def test_reversed_similarity_has_negative_correlation():
    metrics = compute_metrics([0, 0.5, 1], [0.9, 0.5, 0.1], score_min=0, score_max=1)
    assert metrics["spearman"] == pytest.approx(-1)
    assert metrics["pearson"] == pytest.approx(-1)


@dataclass
class TestModelConfig:
    __test__ = False
    name: str = "organization/base"
    revision: str = "base-revision"
    device: str = "cpu"
    max_seq_length: int = 64
    query_prompt: str = "query: "
    document_prompt: str = "passage: "


@dataclass
class TestEvaluationConfig:
    __test__ = False
    dataset: str = "test-fixture"
    batch_size: int = 2
    score_min: float = 0
    score_max: float = 1


@dataclass
class TestConfig:
    __test__ = False
    output_dir: str
    model: TestModelConfig
    evaluation: TestEvaluationConfig


def test_evaluate_preserves_baseline_and_uses_override_without_base_revision(tmp_path, monkeypatch):
    from darijaembed import data

    rows = [
        {"sentence1": "a", "sentence2": "b", "score": 0},
        {"sentence1": "c", "sentence2": "d", "score": 0.5},
        {"sentence1": "e", "sentence2": "f", "score": 1},
    ]
    monkeypatch.setattr(data, "load_source", lambda _: rows)
    instances = []

    class FakeModel:
        def __init__(self, name, *, revision, device):
            self.name, self.revision, self.device = name, revision, device
            self.calls = []
            instances.append(self)

        def encode(self, sentences, **kwargs):
            self.calls.append((sentences, kwargs))
            if sentences == ["a", "c", "e"]:
                return np.array([[1, 0], [1, 0], [1, 0]])
            return np.array([[0, 1], [0.5, np.sqrt(0.75)], [1, 0]])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )
    config = TestConfig(str(tmp_path), TestModelConfig(), TestEvaluationConfig())
    baseline = evaluate(config)
    trained = evaluate(config, model_name="local/model")
    assert baseline["report_path"] != trained["report_path"]
    assert instances[0].revision == "base-revision"
    assert instances[1].revision is None
    assert instances[1].max_seq_length == 64
    assert instances[1].calls[0][1]["prompt"] == "query: "
    assert instances[1].calls[1][1]["prompt"] == "passage: "
    assert instances[1].calls[0][1]["normalize_embeddings"] is True
    assert baseline["dataset_sha256"] == trained["dataset_sha256"]
    assert baseline["metrics"]["spearman"] == pytest.approx(1)
    for report in (baseline, trained):
        with open(report["report_path"], encoding="utf-8") as stream:
            saved = json.load(stream)
        assert saved["metrics"] == report["metrics"]


@pytest.mark.parametrize("score", ["not a score", True, None, float("nan")])
def test_invalid_score_fails_before_model_loading(tmp_path, monkeypatch, score):
    from darijaembed import data

    monkeypatch.setattr(
        data,
        "load_source",
        lambda _: [
            {"sentence1": "a", "sentence2": "b", "score": score},
            {"sentence1": "c", "sentence2": "d", "score": 0.5},
            {"sentence1": "e", "sentence2": "f", "score": 1},
        ],
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    config = TestConfig(str(tmp_path), TestModelConfig(), TestEvaluationConfig())
    with pytest.raises(ValueError):
        evaluate(config)
