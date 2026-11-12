from __future__ import annotations

import numpy as np

from benchmark.legal_eval import (
    evaluate_task_probe,
    fit_task_probe,
    semantic_factor_effects,
)
from runtime.data_loader import LabeledSplit


def _split(name: str, targets: np.ndarray, *, multilabel: bool) -> LabeledSplit:
    count = len(targets)
    width = targets.shape[1]
    ids = np.arange(count * 4, dtype=np.int32).reshape(count, 4)
    return LabeledSplit(
        task_name=name,
        split="test",
        input_ids=ids,
        attention_mask=np.ones_like(ids, dtype=bool),
        transformed_input_ids=ids,
        transformed_attention_mask=np.ones_like(ids, dtype=bool),
        targets=targets,
        label_names=tuple(str(index) for index in range(width)),
        factor_names=tuple(f"{name}:{index}" for index in range(width)),
        factor_mask=targets.astype(bool),
        example_ids=tuple(str(index) for index in range(count)),
        multilabel=multilabel,
        fingerprint=f"{name}-fingerprint",
    )


def test_frozen_multilabel_probe_is_operational() -> None:
    targets = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int8)
    train = _split("eurlex", targets, multilabel=True)
    validation = _split("eurlex", targets, multilabel=True)
    features = np.asarray([[2, 0], [1, 0], [0, 1], [0, 2]], dtype=float)
    probe, _ = fit_task_probe(
        train,
        validation,
        features,
        features,
        alpha=1e-3,
        bins=5,
        acquisition_checkpoint=0,
    )
    metrics, probabilities, losses = evaluate_task_probe(probe, validation, features)
    assert metrics["micro_f1"] == 1.0
    assert probabilities.shape == targets.shape
    assert losses.shape == targets.shape


def test_semantic_effects_use_each_labels_own_loss() -> None:
    factors = np.asarray([[1, 0], [1, 1]], dtype=bool)
    full = np.asarray([[0.1, 0.2], [0.2, 0.3]])
    ablated = np.asarray([[[0.6, 9.0], [0.7, 8.0]]])
    result = semantic_factor_effects(
        full,
        ablated,
        factors,
        min_support=1,
    )
    assert result["signed_effects"][0, 0] == np.mean([0.5, 0.5])
    assert result["signed_effects"][0, 1] == 7.7
