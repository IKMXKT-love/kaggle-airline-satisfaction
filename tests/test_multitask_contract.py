"""Leakage, masking, and objective checks for the denoising experiment."""
import copy
import unittest

import numpy as np
import pandas as pd
import torch

from multitask_support import (CONTEXT, RATINGS, RawEncoder, RatingMLP, batch,
                               combine, conceal, recovery_loss, to_device)


def example():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({c: rng.integers(0, 6, 24) for c in RATINGS})
    frame['Age'] = rng.integers(18, 75, 24)
    frame['Flight Distance'] = rng.integers(50, 2000, 24)
    frame['Departure Delay in Minutes'] = rng.integers(0, 150, 24)
    frame['Arrival Delay in Minutes'] = rng.integers(0, 150, 24).astype(float)
    for c in CONTEXT:
        frame[c] = ['a', 'b'] * 12
    return frame


class Contract(unittest.TestCase):
    def test_fit_local_unknowns_and_missing(self):
        frame = example()
        encoder = RawEncoder().fit(frame.iloc[:16])
        original = copy.deepcopy(encoder.state())
        held = frame.iloc[16:].copy()
        held.loc[held.index[0], 'Flight Distance'] = 99999
        held.loc[held.index[0], 'Gender'] = 'heldout_only'
        held.loc[held.index[0], 'Arrival Delay in Minutes'] = np.nan
        out = encoder.transform(held)
        self.assertEqual(out['cat'][0, encoder.cat_columns.index('Gender')], 0)
        self.assertEqual(out['cat'][0, encoder.cat_columns.index('Flight Distance')], 0)
        self.assertEqual(out['cat'][0, encoder.cat_columns.index('Arrival Delay in Minutes')], 1)
        self.assertEqual(out['missing'][0, encoder.numeric.index('Arrival Delay in Minutes')], 1)
        self.assertEqual(original, encoder.state())
        for forbidden in ['id', 'satisfaction']:
            with self.assertRaises(AssertionError):
                RawEncoder().fit(frame.assign(**{forbidden: 0}))

    def test_mask_hides_all_views_counterfactually(self):
        frame = example()
        frame.loc[0, RATINGS[0]] = 0
        encoder = RawEncoder().fit(frame)
        altered = frame.copy()
        altered.loc[0, RATINGS[0]] = 5
        a = to_device(encoder.transform(frame), 'cpu')
        b = to_device(encoder.transform(altered), 'cpu')
        mask = torch.zeros((len(frame), len(RATINGS)), dtype=torch.bool)
        mask[0, 0] = True
        aa = conceal(batch(a, slice(None)), mask, encoder.rating_numeric, encoder.rating_category)
        bb = conceal(batch(b, slice(None)), mask, encoder.rating_numeric, encoder.rating_category)
        for key in aa:
            torch.testing.assert_close(aa[key], bb[key], rtol=0, atol=0)
        self.assertNotEqual(a['cat'][0, encoder.rating_category[0]], 2)
        self.assertEqual(aa['cat'][0, encoder.rating_category[0]], 2)
        config = dict(members=2, embedding_dim=4, frequencies=2, frequency_sigma=1.,
                      hidden=[16, 8], dropout=0.)
        model = RatingMLP(encoder.state(), config).eval()
        pa, _ = model(batch(a, slice(None)), mask)
        pb, _ = model(batch(b, slice(None)), mask)
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        self.assertNotIn('rating', batch(a, slice(None)))

    def test_recovery_gradient_only_uses_masked_targets(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 4, 13, 6, requires_grad=True)
        target = torch.randint(0, 6, (4, 13))
        mask = torch.zeros(4, 13, dtype=torch.bool)
        mask[1, 5] = True
        original = recovery_loss(logits, target, mask)
        modified = target.clone()
        modified[~mask] = (modified[~mask] + 1) % 6
        torch.testing.assert_close(original, recovery_loss(logits, modified, mask))
        original.backward()
        self.assertEqual(float(logits.grad[:, ~mask, :].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[:, mask, :].abs().sum()), 0.)
        self.assertEqual(float(recovery_loss(logits, target, torch.zeros_like(mask)).detach()), 0.)

    def test_frozen_fusion_identity(self):
        a = np.array([.01, .3, .99])
        b = np.array([.1, .4, .8])
        current = combine(a, b)
        np.testing.assert_allclose(combine(a, b, b, 'replace35'), current, atol=1e-15)
        np.testing.assert_allclose(combine(a, b, current, 'add20'), current, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
