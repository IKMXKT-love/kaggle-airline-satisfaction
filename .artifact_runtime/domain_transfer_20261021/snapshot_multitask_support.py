"""Fold-local raw features and a vectorized ensemble for rating denoising."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
CONTEXT = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
RATINGS = ['Inflight wifi service', 'Departure/Arrival time convenient',
           'Ease of Online booking', 'Gate location', 'Food and drink',
           'Online boarding', 'Seat comfort', 'Inflight entertainment',
           'On-board service', 'Leg room service', 'Baggage handling',
           'Checkin service', 'Cleanliness']
SPECS = ['A_clean', 'B_masked', 'C_multitask']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_torch(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def cpu_state(model):
    return {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}


class RawEncoder:
    """No label-dependent features, no state changes on transform.

    Categorical codes: unknown=0, true missing=1, artificial mask=2,
    known values start at 3. Numerical views never replace real rating zero
    with the mask code. All vocabularies/scales are fit on fit rows only.
    """
    @staticmethod
    def tokens(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.astype('float64').astype('string')
        return s.astype('string')

    def numbers(self, frame):
        a = frame[self.numeric].astype('float64').to_numpy(copy=True)
        for j, col in enumerate(self.numeric):
            if col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
                assert np.all(a[np.isfinite(a[:, j]), j] >= 0)
                a[:, j] = np.log1p(a[:, j])
        return a

    def fit(self, frame):
        assert not {'id', 'satisfaction'} & set(frame)
        assert set(CONTEXT + RATINGS).issubset(frame)
        self.columns = list(frame)
        assert len(self.columns) == 21
        self.numeric = [c for c in self.columns if c not in CONTEXT]
        self.cat_columns = CONTEXT + self.numeric
        self.rating_numeric = [self.numeric.index(c) for c in RATINGS]
        self.rating_category = [self.cat_columns.index(c) for c in RATINGS]
        a = self.numbers(frame)
        self.medians = np.nanmedian(a, axis=0)
        self.medians = np.where(np.isfinite(self.medians), self.medians, 0.)
        q = np.nanquantile(a, [.25, .75], axis=0)
        self.scales = np.maximum(q[1] - q[0], 1.)
        self.scales = np.where(np.isfinite(self.scales), self.scales, 1.)
        self.vocab = {c: sorted(self.tokens(frame[c]).dropna().unique().tolist())
                      for c in self.cat_columns}
        self.cat_sizes = [len(self.vocab[c]) + 3 for c in self.cat_columns]
        return self

    def transform(self, frame):
        assert list(frame) == self.columns
        assert not {'id', 'satisfaction'} & set(frame)
        a = self.numbers(frame)
        missing = (~np.isfinite(a)).astype('float32')
        a = np.where(np.isfinite(a), a, self.medians)
        a = 3 * np.tanh(((a - self.medians) / self.scales) / 3)
        categories = []
        for c in self.cat_columns:
            tokens = self.tokens(frame[c])
            table = {value: i + 3 for i, value in enumerate(self.vocab[c])}
            ids = tokens.map(table).fillna(0).to_numpy(dtype='int64', copy=True)
            ids[tokens.isna().to_numpy()] = 1
            categories.append(ids)
        target = frame[RATINGS].to_numpy(dtype='float64')
        assert np.isfinite(target).all(), 'Rating missingness requires a separate recovery mask'
        assert ((target >= 0) & (target <= 5) & (target == np.floor(target))).all()
        return dict(num=a.astype('float32'), missing=missing,
                    cat=np.column_stack(categories), rating=target.astype('int64'))

    def state(self):
        return dict(columns=self.columns, numeric=self.numeric, cat_columns=self.cat_columns,
                    rating_numeric=self.rating_numeric, rating_category=self.rating_category,
                    medians=self.medians.tolist(), scales=self.scales.tolist(),
                    vocab=self.vocab, cat_sizes=self.cat_sizes)


def to_device(values, device):
    return {k: torch.as_tensor(v, device=device) for k, v in values.items()}


def batch(values, index):
    # Recovery targets are deliberately not returned to the forward path.
    return {k: values[k][index] for k in ['num', 'missing', 'cat']}


def conceal(inputs, mask, rating_numeric, rating_category):
    out = {k: v.clone() for k, v in inputs.items()}
    rn = torch.as_tensor(rating_numeric, device=mask.device)
    rc = torch.as_tensor(rating_category, device=mask.device)
    out['num'][:, rn] = torch.where(mask, 0., out['num'][:, rn])
    out['missing'][:, rn] = torch.where(mask, 0., out['missing'][:, rn])
    out['cat'][:, rc] = torch.where(mask, 2, out['cat'][:, rc])
    return out


class EnsembleLinear(nn.Module):
    def __init__(self, members, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, n_in, n_out))
        self.bias = nn.Parameter(torch.empty(members, 1, n_out))
        bound = 1 / math.sqrt(n_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias


class RatingMLP(nn.Module):
    """Independent parameters per ensemble member; shared minibatch rows.

    This is a standalone MLP with numerical embeddings, not the pytabkit
    RealMLP implementation. Every A/B/C model includes identical heads.
    """
    def __init__(self, encoder_state, config, prior=.5):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.members = config['members']
        self.n_numeric = len(encoder_state['numeric'])
        self.rating_numeric = encoder_state['rating_numeric']
        self.rating_category = encoder_state['rating_category']
        sizes = encoder_state['cat_sizes']
        self.n_categories = sum(sizes)
        offsets = np.cumsum([0] + sizes[:-1])
        self.register_buffer('cat_offsets', torch.tensor(offsets, dtype=torch.long))
        self.register_buffer('ensemble_offsets', torch.arange(self.members)[:, None, None] * self.n_categories)
        self.embedding = nn.Embedding(self.members * self.n_categories, config['embedding_dim'])
        nn.init.normal_(self.embedding.weight, std=.1)
        self.frequency = nn.Parameter(torch.randn(self.members, self.n_numeric, config['frequencies'])
                                      * config['frequency_sigma'])
        n_in = (len(sizes) * config['embedding_dim']
                + self.n_numeric * (2 + 2 * config['frequencies']) + len(RATINGS))
        dimensions = [n_in] + config['hidden']
        self.layers = nn.ModuleList([EnsembleLinear(self.members, a, b)
                                     for a, b in zip(dimensions[:-1], dimensions[1:])])
        self.classifier = EnsembleLinear(self.members, dimensions[-1], 1)
        self.recovery = EnsembleLinear(self.members, dimensions[-1], len(RATINGS) * 6)
        nn.init.normal_(self.classifier.weight, std=.01)
        nn.init.constant_(self.classifier.bias, float(logit(np.clip(prior, 1e-6, 1 - 1e-6))))

    def forward(self, inputs, mask=None, recovery=False):
        size = inputs['num'].shape[0]
        if mask is None:
            mask = torch.zeros((size, len(RATINGS)), device=inputs['num'].device, dtype=torch.bool)
        visible = conceal(inputs, mask, self.rating_numeric, self.rating_category)
        ids = visible['cat'][None, :, :] + self.cat_offsets[None, None, :] + self.ensemble_offsets
        categories = self.embedding(ids).flatten(2)
        num = visible['num'][None, :, :].expand(self.members, -1, -1)
        phase = 2 * math.pi * num[:, :, :, None] * self.frequency[:, None, :, :]
        periodic = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(2)
        missing = visible['missing'][None, :, :].expand(self.members, -1, -1)
        flags = mask.float()[None, :, :].expand(self.members, -1, -1)
        z = torch.cat([num, missing, periodic, categories, flags], dim=-1)
        for layer in self.layers:
            z = F.dropout(F.silu(layer(z)), p=self.config['dropout'], training=self.training)
        prediction = self.classifier(z).squeeze(-1)
        decoded = self.recovery(z).reshape(self.members, size, len(RATINGS), 6) if recovery else None
        return prediction, decoded


def classification_loss(logits, target, smoothing=.01):
    target = target.float() * (1 - smoothing) + .5 * smoothing
    return F.binary_cross_entropy_with_logits(logits.float(), target[None, :].expand_as(logits))


def recovery_loss(logits, target, mask):
    if not mask.any():
        return logits.float().sum() * 0
    chosen = logits[:, mask, :].float()
    labels = target[mask][None, :].expand(logits.shape[0], -1)
    return F.cross_entropy(chosen.reshape(-1, 6), labels.reshape(-1))


@torch.no_grad()
def predict(model, values, batch_size, amp=True):
    model.eval()
    outputs = []
    device = values['num'].device
    for start in range(0, len(values['num']), batch_size):
        idx = slice(start, start + batch_size)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(batch(values, idx))
        outputs.append(torch.sigmoid(logits.float()).mean(0).cpu().numpy())
    return np.concatenate(outputs).astype('float64')


@torch.no_grad()
def reconstruction_metrics(model, values, batch_size, seed=1942, probability=.15, amp=True):
    model.eval()
    device = values['num'].device
    generator = torch.Generator(device=device).manual_seed(seed)
    loss_sum, correct_sum, positions = 0., 0., 0
    for start in range(0, len(values['num']), batch_size):
        idx = slice(start, start + batch_size)
        count = len(values['num'][idx])
        mask = torch.rand((count, len(RATINGS)), device=device, generator=generator) < probability
        if not mask.any():
            continue
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            _, decoded = model(batch(values, idx), mask, recovery=True)
        n = int(mask.sum())
        loss_sum += float(recovery_loss(decoded, values['rating'][idx], mask)) * n
        probabilities = decoded.float().softmax(-1).mean(0)
        correct_sum += int((probabilities.argmax(-1)[mask] == values['rating'][idx][mask]).sum())
        positions += n
    return dict(cross_entropy=loss_sum / positions, accuracy=correct_sum / positions,
                masked_positions=positions, seed=seed)


def combine(v4, lean, candidate=None, usage='current'):
    def margin(p):
        return logit(np.clip(p, 1e-6, 1 - 1e-6))
    current = .65 * margin(v4) + .35 * margin(lean)
    if usage == 'current':
        return expit(current)
    if usage == 'replace35':
        return expit(.65 * margin(v4) + .35 * margin(candidate))
    if usage == 'add20':
        return expit(.8 * current + .2 * margin(candidate))
    raise ValueError(usage)


def paired(y, p, q):
    def placements(score):
        pos, neg = score[y == 1], score[y == 0]
        sp, sn = np.sort(pos), np.sort(neg)
        vp = (np.searchsorted(sn, pos, side='left') + np.searchsorted(sn, pos, side='right')) / (2 * len(neg))
        vn = 1 - (np.searchsorted(sp, neg, side='left') + np.searchsorted(sp, neg, side='right')) / (2 * len(pos))
        assert abs(vp.mean() - roc_auc_score(y, score)) < 1e-12
        return vp, vn
    pp, pn = placements(p)
    qp, qn = placements(q)
    dp, dn = pp - qp, pn - qn
    delta = float(dp.mean())
    se = float(np.sqrt(dp.var(ddof=1) / len(dp) + dn.var(ddof=1) / len(dn)))
    return dict(delta=delta, approximate_se=se, interval95=[delta - 1.96 * se, delta + 1.96 * se])
