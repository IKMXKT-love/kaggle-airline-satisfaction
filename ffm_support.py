"""Small field-aware classifier with a matching additive-only control."""
import numpy as np
import pandas as pd
import torch
from torch import nn

CONTINUOUS = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']


class FieldEncoder:
    @staticmethod
    def token(s):
        return s.astype('string')

    def fit(self, x):
        assert not {'id', 'satisfaction'} & set(x)
        self.columns = list(x)
        self.vocab = {c: {s: k+2 for k, s in enumerate(sorted(self.token(x[c]).dropna().unique()))}
                      for c in x}
        self.sizes = [len(self.vocab[c])+2 for c in x]
        self.offsets = np.r_[0, np.cumsum(self.sizes)[:-1]].astype('int64')
        raw = self.numbers(x)
        self.median = np.nanmedian(raw, axis=0)
        self.iqr = np.maximum(np.nanquantile(raw, .75, axis=0)-np.nanquantile(raw, .25, axis=0), 1.)
        return self

    @staticmethod
    def numbers(x):
        a = x[CONTINUOUS].to_numpy(dtype='float64', copy=True)
        a[:, 2:] = np.log1p(a[:, 2:])
        return a

    def transform(self, x):
        assert list(x) == self.columns and not {'id', 'satisfaction'} & set(x)
        tokens = []
        for j, c in enumerate(x):
            s = self.token(x[c])
            a = s.map(self.vocab[c]).fillna(1).to_numpy(dtype='int64', copy=True)
            a[s.isna().to_numpy()] = 0
            tokens.append(a+self.offsets[j])
        a = self.numbers(x)
        missing = np.isnan(a).astype('float32')
        a = np.clip(np.nan_to_num((a-self.median)/self.iqr), -6, 6)
        hinge = np.maximum(a[:, :, None]-np.linspace(-3, 3, 9), 0).reshape(len(x), -1)
        numeric = np.column_stack([a, missing, hinge]).astype('float32')
        return dict(tokens=np.column_stack(tokens), numeric=numeric)


class FieldModel(nn.Module):
    def __init__(self, total_tokens, fields, numeric_dim, interactions, members=4, rank=8):
        super().__init__()
        self.fields = fields
        self.members = members
        self.main = nn.Parameter(torch.zeros(members, total_tokens))
        self.numeric = nn.Parameter(torch.randn(members, numeric_dim)*.01)
        self.bias = nn.Parameter(torch.zeros(members))
        self.interactions = interactions
        if interactions:
            self.factors = nn.Parameter(torch.randn(members, total_tokens*fields, rank)*.02)
        left, right = torch.triu_indices(fields, fields, offset=1)
        self.register_buffer('left', left)
        self.register_buffer('right', right)
        self.register_buffer('member_idx', torch.arange(members)[:, None, None])

    def forward(self, tokens, numeric):
        out = self.main[:, tokens].sum(-1) + self.numeric @ numeric.T + self.bias[:, None]
        if self.interactions:
            left = tokens[:, self.left]*self.fields+self.right
            right = tokens[:, self.right]*self.fields+self.left
            a = self.factors[self.member_idx, left[None, :, :], :]
            b = self.factors[self.member_idx, right[None, :, :], :]
            out = out + (a*b).sum((-1, -2))
        return out


def tensor_data(data, device):
    return {k: torch.as_tensor(v, device=device) for k, v in data.items()}


@torch.no_grad()
def predict(model, data, batch_size=4096):
    model.eval()
    values = []
    for start in range(0, len(data['tokens']), batch_size):
        values.append(model(data['tokens'][start:start+batch_size],
                            data['numeric'][start:start+batch_size]).sigmoid().mean(0).float().cpu().numpy())
    return np.concatenate(values)
