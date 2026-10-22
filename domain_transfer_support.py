"""Shared raw-field representation with separately selectable domain heads."""
import copy
import math

import numpy as np
import torch
from scipy.special import logit
from torch import nn
from torch.nn import functional as F

from multitask_support import RawEncoder, EnsembleLinear, batch, to_device

ARMS = ['competition_only', 'shared_head', 'dual_head']


class DomainMLP(nn.Module):
    """Independent ensemble members; shared encoder across the two data domains.

    Each arm has the same parameters and initialization. No cross-row attention,
    batch statistics, teacher predictions, or label inputs occur in forward().
    """
    def __init__(self, encoder_state, config, prior):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.members = config['members']
        self.n_numeric = len(encoder_state['numeric'])
        sizes = encoder_state['cat_sizes']
        self.n_categories = sum(sizes)
        self.register_buffer('cat_offsets', torch.tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long))
        self.register_buffer('ensemble_offsets', torch.arange(self.members)[:, None, None] * self.n_categories)
        self.embedding = nn.Embedding(self.members * self.n_categories, config['embedding_dim'])
        nn.init.normal_(self.embedding.weight, std=.1)
        self.frequency = nn.Parameter(torch.randn(self.members, self.n_numeric, config['frequencies'])
                                      * config['frequency_sigma'])
        n_in = len(sizes) * config['embedding_dim'] + self.n_numeric * (2 + 2 * config['frequencies'])
        dims = [n_in] + config['hidden']
        self.layers = nn.ModuleList([EnsembleLinear(self.members, a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.competition_head = EnsembleLinear(self.members, dims[-1], 1)
        nn.init.normal_(self.competition_head.weight, std=.01)
        nn.init.constant_(self.competition_head.bias, float(logit(np.clip(prior, 1e-6, 1-1e-6))))
        self.original_head = copy.deepcopy(self.competition_head)

    def forward(self, inputs):
        ids = inputs['cat'][None] + self.cat_offsets[None, None] + self.ensemble_offsets
        categories = self.embedding(ids).flatten(2)
        num = inputs['num'][None].expand(self.members, -1, -1)
        phase = 2 * math.pi * num[:, :, :, None] * self.frequency[:, None]
        periodic = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(2)
        missing = inputs['missing'][None].expand(self.members, -1, -1)
        hidden = torch.cat([num, missing, periodic, categories], dim=-1)
        for layer in self.layers:
            hidden = F.dropout(F.silu(layer(hidden)), p=self.config['dropout'], training=self.training)
        return self.competition_head(hidden).squeeze(-1), self.original_head(hidden).squeeze(-1)


def domain_loss(outputs, y_competition, y_original, arm, source_weight=.25, smoothing=.005):
    assert arm in ARMS
    a, b = outputs
    n = len(y_competition)
    yc = y_competition.float() * (1-smoothing) + .5*smoothing
    yo = y_original.float() * (1-smoothing) + .5*smoothing
    lc = F.binary_cross_entropy_with_logits(a[:, :n].float(), yc[None].expand_as(a[:, :n]))
    original_logits = b[:, n:] if arm == 'dual_head' else a[:, n:]
    lo = F.binary_cross_entropy_with_logits(original_logits.float(), yo[None].expand_as(original_logits))
    # Both losses are separately normalized over rows and ensemble members.
    return lc + (0. if arm == 'competition_only' else source_weight) * lo, lc.detach(), lo.detach()


@torch.no_grad()
def predict(model, values, batch_size=4096):
    model.eval()
    out = []
    for start in range(0, len(values['num']), batch_size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch(values, slice(start, start+batch_size)))
        out.append(logits.float().sigmoid().mean(0).cpu().numpy())
    return np.concatenate(out).astype('float64')


def joint_batch(competition, original, ci, oi):
    return {k: torch.cat([competition[k][ci], original[k][oi]], dim=0)
            for k in ['num', 'missing', 'cat']}
