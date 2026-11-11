"""Optional real Trainer check using random local weights and synthetic text."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from darijaembed.cli import _prepare
from darijaembed.config import load_config
from darijaembed.evaluation import evaluate
from darijaembed.training import train


@pytest.mark.skipif(
    os.environ.get("DARIJAEMBED_RUN_ML_TESTS") != "1",
    reason="Set DARIJAEMBED_RUN_ML_TESTS=1 and install the ML extras.",
)
def test_real_prepare_evaluate_train_and_reload(tmp_path):
    import torch
    from sentence_transformers import SentenceTransformer, models
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.set_num_threads(2)
    torch.manual_seed(42)
    repository = Path(__file__).resolve().parents[1]
    fixtures = repository / "tests/fixtures"
    texts = (fixtures / "pairs.jsonl").read_text(encoding="utf-8")
    texts += (fixtures / "sts.jsonl").read_text(encoding="utf-8")
    characters = sorted({character for character in texts if character.isalnum()})
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens += characters + ["##" + character for character in characters]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    backend = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    backbone = tmp_path / "backbone"
    tokenizer.save_pretrained(backbone)
    BertModel(
        BertConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=256,
        )
    ).save_pretrained(backbone)
    transformer = models.Transformer(str(backbone), max_seq_length=64)
    model = SentenceTransformer(modules=[transformer, models.Pooling(16)], device="cpu")
    baseline = tmp_path / "baseline"
    model.save_pretrained(baseline)
    original = load_config(repository / "configs/local.toml")
    config = replace(
        original,
        model=replace(original.model, name=str(baseline), max_seq_length=64),
        output_dir=str(tmp_path / "run"),
    )

    preparation = _prepare(config)
    before = evaluate(config)
    trained = train(config, prepared_dir=Path(config.output_dir) / "data")
    after = evaluate(config, model_name=str(Path(config.output_dir) / "model"))

    assert preparation["train_rows"] == 6
    assert preparation["validation_rows"] == 2
    assert trained["status"] == "completed"
    assert trained["train_metrics"]["train_loss"] > 0
    assert trained["validation_metrics"]["eval_loss"] > 0
    assert before["report_path"] != after["report_path"]
    assert Path(before["report_path"]).exists()
    assert Path(after["report_path"]).exists()
