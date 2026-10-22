"""Class-conditional domain odds, with label-free query inference."""
import numpy as np
import torch
from torch.nn import functional as F

from domain_transfer_support import DomainMLP
from multitask_support import batch

ARMS = ['direct', 'teacher_offset', 'conditional_ratio']
LIMIT = 8.


class RatioMLP(DomainMLP):
    def __init__(self, state, config):
        super().__init__(state, config, .5)
        # The common domain component and class contrast both start at zero.
        for head in [self.competition_head, self.original_head]:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, inputs):
        common, contrast = super().forward(inputs)
        return common.float(), LIMIT * torch.tanh(contrast.float() / LIMIT)


def domain_logits(outputs, labels):
    common, contrast = outputs
    return common + (2 * labels[None].float() - 1) * contrast / 2


def query_logits(outputs, teacher, arm, competition_prior_logit):
    # Neither observed query class nor domain indicator is an inference input.
    _, contrast = outputs
    if arm == 'direct':
        return contrast + competition_prior_logit
    return contrast + teacher[None]


def loss_for_arm(outputs, y, domain, teacher, arm, competition_prior):
    if arm == 'conditional_ratio':
        margins = domain_logits(outputs, y)
        return F.binary_cross_entropy_with_logits(margins, domain[None].expand_as(margins))
    mask = domain.bool()
    margins = query_logits(outputs, teacher, arm, float(np.log(competition_prior / (1-competition_prior))))[:, mask]
    target = y[mask].float()
    # C0 and C1 are sampled equally. Restore the natural competition class prior.
    weight = torch.where(target.bool(), 2*competition_prior, 2*(1-competition_prior))
    return (F.binary_cross_entropy_with_logits(margins, target[None].expand_as(margins), reduction='none') * weight[None]).mean()


def draw_task(c, o, yc, yo, pools, teacher, generator, group_size):
    # Equal C0,C1,O0,O1; independent sampling with replacement, reproducible RNG.
    indices = [p[torch.randint(len(p), (group_size,), device=p.device, generator=generator)] for p in pools]
    ci, oi = torch.cat(indices[:2]), torch.cat(indices[2:])
    inputs = {k: torch.cat([c[k][ci], o[k][oi]]) for k in ['num','missing','cat']}
    y = torch.cat([yc[ci], yo[oi]])
    domain = torch.cat([torch.ones(len(ci), device=ci.device), torch.zeros(len(oi), device=oi.device)])
    # No original-domain teacher values are computed or exposed to the network.
    q = torch.cat([teacher[ci], torch.zeros(len(oi), device=ci.device)])
    return inputs, y, domain, q


@torch.no_grad()
def predict(model, values, teacher, arm, prior_logit, batch_size=4096):
    model.eval()
    probabilities, contrasts = [], []
    for start in range(0, len(values['num']), batch_size):
        rows = slice(start, start+batch_size)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(batch(values, rows))
        probabilities.append(query_logits(outputs, teacher[rows], arm, prior_logit).sigmoid().mean(0).cpu().numpy())
        contrasts.append(outputs[1].mean(0).cpu().numpy())
    return np.concatenate(probabilities).astype('float64'), np.concatenate(contrasts).astype('float64')
