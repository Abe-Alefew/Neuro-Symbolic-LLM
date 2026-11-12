from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from runtime import data_loader


@dataclass
class _ClassLabel:
    names: list[str]


@dataclass
class _Sequence:
    feature: _ClassLabel


class _Dataset(list):
    def __init__(self, rows, feature):
        super().__init__(rows)
        self.column_names = list(rows[0])
        self.features = {"text": object(), "labels": feature, "label": feature}


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        length = kwargs["max_length"]
        values = [min(ord(char), 255) for char in text][:length]
        mask = [1] * len(values)
        values += [0] * (length - len(values))
        mask += [0] * (length - len(mask))
        return {
            "input_ids": np.asarray([values], dtype=np.int32),
            "attention_mask": np.asarray([mask], dtype=np.int32),
        }


def _config(multilabel: bool) -> dict:
    return {
        "name": "eurlex" if multilabel else "scotus",
        "dataset": "coastalcph/lex_glue",
        "config": "eurlex" if multilabel else "scotus",
        "revision": "abc123",
        "text_field": "text",
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "legal_task": {
            "label_field": "labels" if multilabel else "label",
            "multilabel": multilabel,
            "expected_num_labels": 3,
        },
    }


def test_loads_multilabel_targets_and_semantic_factors(monkeypatch):
    dataset = _Dataset(
        [{"text": "alpha", "labels": [0, 2]}, {"text": "beta", "labels": [1]}],
        _Sequence(_ClassLabel(["a", "b", "c"])),
    )
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    monkeypatch.setattr(data_loader, "load_dataset_split", load)
    spec = data_loader.parse_legal_task_spec(_config(True))
    assert spec is not None
    split = data_loader.load_labeled_split(
        spec, "test", _Tokenizer(), sequence_length=8
    )
    np.testing.assert_array_equal(split.targets, [[1, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(split.factor_mask, split.targets.astype(bool))
    assert split.factor_names == ("eurlex:a", "eurlex:b", "eurlex:c")
    assert len(split.fingerprint) == 64
    assert calls[0][1]["revision"] == "abc123"
    assert calls[0][0][1] == "test"


def test_loads_single_label_targets(monkeypatch):
    dataset = _Dataset(
        [{"text": "one", "label": 2}, {"text": "two", "label": 0}],
        _ClassLabel(["x", "y", "z"]),
    )
    monkeypatch.setattr(
        data_loader, "load_dataset_split", lambda *args, **kwargs: dataset
    )
    spec = data_loader.parse_legal_task_spec(_config(False))
    assert spec is not None
    split = data_loader.load_labeled_split(
        spec, "validation", _Tokenizer(), sequence_length=6
    )
    np.testing.assert_array_equal(split.probe_targets(), [2, 0])
    assert split.multilabel is False


def test_legal_task_requires_pinned_revision():
    config = _config(False)
    del config["revision"]
    with pytest.raises(ValueError, match="pinned revision"):
        data_loader.parse_legal_task_spec(config)
