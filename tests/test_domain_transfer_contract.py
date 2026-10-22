"""Gradient isolation and domain-loss weighting contracts."""
import copy
import unittest

import torch

from domain_transfer_support import DomainMLP, domain_loss


class DomainContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.model = DomainMLP({'numeric': ['x'], 'cat_sizes': [9]},
            dict(members=2, embedding_dim=3, frequencies=2, frequency_sigma=1., hidden=[12, 8], dropout=0.), .45)
        self.x = {'num': torch.randn(10, 1), 'missing': torch.zeros(10, 1),
                  'cat': torch.arange(10).remainder(9).reshape(-1, 1)}
        self.yc = torch.tensor([0., 1., 0., 1., 0., 1.])
        self.yo = torch.tensor([0., 1., 1., 0.])

    def gradients(self, arm, original):
        self.model.zero_grad(set_to_none=True)
        loss, _, _ = domain_loss(self.model(self.x), self.yc, original, arm)
        loss.backward()
        return {n: None if p.grad is None else p.grad.detach().clone() for n, p in self.model.named_parameters()}

    def test_original_labels_cannot_change_competition_only_gradients(self):
        a = self.gradients('competition_only', self.yo)
        b = self.gradients('competition_only', 1-self.yo)
        for key in a:
            if a[key] is None:
                self.assertIsNone(b[key])
            else:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_separate_head_isolates_source_head_gradient_but_updates_encoder(self):
        a = self.gradients('dual_head', self.yo)
        b = self.gradients('dual_head', 1-self.yo)
        torch.testing.assert_close(a['competition_head.weight'], b['competition_head.weight'], rtol=0, atol=0)
        self.assertGreater(float((a['original_head.weight']-b['original_head.weight']).abs().max()), 0.)
        self.assertGreater(float((a['layers.0.weight']-b['layers.0.weight']).abs().max()), 0.)
        c = self.gradients('shared_head', self.yo)
        d = self.gradients('shared_head', 1-self.yo)
        self.assertGreater(float((c['competition_head.weight']-d['competition_head.weight']).abs().max()), 0.)

    def test_source_weight_does_not_depend_on_original_batch_repetition(self):
        outputs = self.model(self.x)
        for arm in ['shared_head', 'dual_head']:
            expected = domain_loss(outputs, self.yc, self.yo, arm)[0]
            doubled = tuple(torch.cat([v[:, :6], v[:, 6:], v[:, 6:]], 1) for v in outputs)
            actual = domain_loss(doubled, self.yc, self.yo.repeat(2), arm)[0]
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
