"""Contracts for the resource adapter and isolation of context/query rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import unittest
from unittest.mock import patch
from tabicl_finetune_support import BoundedFinetuner, SequentialViewsClassifier
from tabicl._finetune.data import _chunk_indices, _split_ctx_query


def check_bounded_fit():
    captured = []

    def recording_fit(self, x, y):
        captured.append((x.copy(), y.copy()))
        return self

    trainer = BoundedFinetuner(model_path='unused.ckpt')
    trainer.training_rows_ = 8
    trainer.context_positions_ = np.array([6, 2, 5])
    with patch.object(SequentialViewsClassifier, 'fit', recording_fit):
        inner = trainer._create_inner_estimator(n_estimators=4, device='cpu')
        x = np.arange(24).reshape(8, 3)
        y = np.arange(8)
        inner.fit(x, y)
        np.testing.assert_array_equal(captured[0][0], x[[6, 2, 5]])
        np.testing.assert_array_equal(captured[0][1], y[[6, 2, 5]])
        with unittest.TestCase().assertRaisesRegex(AssertionError, 'row order/size'):
            inner.fit(x[:7], y[:7])


def check_chunk_coverage(rows):
    for seed in [42, 43, 44]:
        chunks = _chunk_indices(rows, 2048, np.random.default_rng(seed))
        indices = np.concatenate(chunks)
        assert len(indices) == rows
        np.testing.assert_array_equal(np.sort(indices), np.arange(rows))


def check_context_query():
    y = np.tile([0, 1], 1024)
    context, query = _split_ctx_query(y, query_size=410, seed=42, stratify=True)
    assert len(context) == 1638 and len(query) == 410
    assert not set(context) & set(query)
    np.testing.assert_array_equal(np.sort(np.r_[context, query]), np.arange(2048))
    assert set(y[context]) == set(y[query]) == {0, 1}


class ContractTests(unittest.TestCase):
    def test_bounded_context(self):
        check_bounded_fit()

    def test_screen_coverage(self):
        check_chunk_coverage(411944)

    def test_confirmation_coverage(self):
        check_chunk_coverage(514931)

    def test_context_query_partition(self):
        check_context_query()
