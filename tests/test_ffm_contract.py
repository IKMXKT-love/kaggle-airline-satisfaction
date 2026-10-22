import copy
import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from ffm_support import FieldEncoder, FieldModel


class Contract(unittest.TestCase):
    def test_fit_local_encoder_and_distinct_zero_missing_unknown(self):
        x = pd.DataFrame(dict(Age=[20, 30, 40], **{'Flight Distance': [100, 200, 300],
            'Departure Delay in Minutes': [0., 2., 3.], 'Arrival Delay in Minutes': [0., np.nan, 3.]}))
        enc = FieldEncoder().fit(x)
        before = copy.deepcopy(enc.vocab)
        q = x.iloc[[0, 1]].copy(); q.loc[q.index[1], 'Arrival Delay in Minutes'] = 9999
        tokens = enc.transform(q)['tokens']
        self.assertNotEqual(tokens[0, -1], tokens[1, -1])
        self.assertNotEqual(enc.transform(x)['tokens'][1, -1], tokens[1, -1])
        self.assertEqual(before, enc.vocab)
        with self.assertRaises(AssertionError):
            FieldEncoder().fit(x.assign(satisfaction=0))

    def test_interactions_can_learn_xor_that_additive_control_cannot(self):
        torch.set_num_threads(2)
        tokens = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3]]*64)
        target = torch.tensor([0., 1., 1., 0.]*64)
        n = torch.empty(len(tokens), 0)
        outcomes = []
        for interaction in [False, True]:
            torch.manual_seed(21)
            model = FieldModel(4, 2, 0, interaction, members=1, rank=4)
            opt = torch.optim.Adam(model.parameters(), lr=.03)
            for _ in range(150):
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(tokens, n).squeeze(0), target)
                loss.backward(); opt.step()
            outcomes.append(float(loss.detach()))
        self.assertGreater(outcomes[0], .68)
        self.assertLess(outcomes[1], .05)


if __name__ == '__main__':
    unittest.main()
