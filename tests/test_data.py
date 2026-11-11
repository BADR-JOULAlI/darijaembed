"""Check data quality and leakage safeguards without model downloads."""

from types import SimpleNamespace

import pytest

from darijaembed.data import load_source, normalize_text, prepare_pairs, read_jsonl, write_jsonl


def texts(rows):
    return {text for row in rows for text in row.values()}


def test_split_keeps_transitive_links_and_negatives_together():
    rows = [
        {"anchor": "a", "positive": "b", "negative": "c"},
        {"anchor": "d", "positive": "e", "negative": "c"},
        {"anchor": "e", "positive": "f", "negative": "g"},
        {"anchor": "h", "positive": "i", "negative": "j"},
        {"anchor": "k", "positive": "l", "negative": "m"},
    ]
    train, validation, audit = prepare_pairs(rows, 0.4, 42)
    assert train and validation
    assert texts(train).isdisjoint(texts(validation))
    assert audit["components"] == 3
    assert len(train) + len(validation) == len(rows)
    first_component = {"a", "b", "c", "d", "e", "f", "g"}
    assert first_component.issubset(texts(train)) or first_component.issubset(texts(validation))


def test_cleaning_reports_invalid_duplicates_and_preserves_language():
    rows = [
        {"anchor": " بغيت  نشري\nطوموبيل ", "positive": "Bghit nchri tomobil"},
        {"anchor": "Bghit nchri tomobil", "positive": "بغيت نشري طوموبيل"},
        {"anchor": "same", "positive": "same"},
        {"anchor": "blank", "positive": "  "},
        {"anchor": 12, "positive": "word"},
        {"anchor": "bad", "positive": "negative", "negative": " bad "},
        {"anchor": "another", "positive": "pair"},
    ]
    train, validation, audit = prepare_pairs(rows, 0.2, 7)
    assert audit["input_rows"] == 7
    assert audit["valid_rows"] == 2
    assert audit["invalid_rows"] == 3
    assert audit["duplicate_rows"] == 1
    assert audit["self_pairs"] == 1
    assert "بغيت نشري طوموبيل" in texts(train + validation)
    assert normalize_text(" cafe\u0301 \t ") == "café"


def test_split_is_reproducible():
    rows = [{"anchor": f"a{i}", "positive": f"p{i}"} for i in range(20)]
    assert prepare_pairs(rows, 0.2, 42) == prepare_pairs(rows, 0.2, 42)
    assert prepare_pairs(rows, 0.2, 42)[1] != prepare_pairs(rows, 0.2, 43)[1]


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_invalid_validation_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        prepare_pairs([], fraction, 42)


def test_connected_dataset_cannot_be_split():
    with pytest.raises(ValueError, match="disconnected"):
        prepare_pairs(
            [{"anchor": "a", "positive": "b"}, {"anchor": "b", "positive": "c"}],
            0.2,
            42,
        )


def test_mixed_pairs_and_triplets_are_rejected():
    with pytest.raises(ValueError, match="pairs or triplets"):
        prepare_pairs(
            [{"anchor": "a", "positive": "b"}, {"anchor": "c", "positive": "d", "negative": "e"}],
            0.2,
            42,
        )


def config(path, columns, **kwargs):
    return SimpleNamespace(
        path=str(path),
        columns=columns,
        max_samples=kwargs.get("max_samples"),
        format=kwargs.get("format"),
        name=None,
        split="train",
        revision=None,
    )


def test_jsonl_mapping_limit_and_unicode_roundtrip(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [{"ar": "بغيت", "latin": "bghit"}, {"ar": "سلام", "latin": "salam"}])
    assert read_jsonl(path)[0]["ar"] == "بغيت"
    loaded = load_source(config(path, {"anchor": "ar", "positive": "latin"}, max_samples=1))
    assert loaded == [{"anchor": "بغيت", "positive": "bghit"}]


def test_csv_mapping_and_missing_column(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text('first,second\n"hello, there",salam\n', encoding="utf-8")
    assert load_source(config(path, {"anchor": "first", "positive": "second"})) == [
        {"anchor": "hello, there", "positive": "salam"}
    ]
    with pytest.raises(ValueError, match="missing configured columns"):
        load_source(config(path, {"anchor": "missing"}))


def test_jsonl_errors_include_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\ninvalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        read_jsonl(path)
