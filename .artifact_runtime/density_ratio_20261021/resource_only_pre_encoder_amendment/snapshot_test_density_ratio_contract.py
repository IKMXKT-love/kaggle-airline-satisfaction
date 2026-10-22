import unittest
import numpy as np
import torch
from density_ratio_support import RatioMLP, domain_logits, query_logits, loss_for_arm


class DensityContract(unittest.TestCase):
    def test_balanced_domain_odds_identity(self):
        # Known normalized class-conditional densities, two feature states.
        original = np.array([[.8,.2],[.3,.7]])
        competition = np.array([[.6,.4],[.1,.9]])
        ratio = np.log(competition/original)
        source = np.log(original[1]/original[0])
        np.testing.assert_allclose(source+ratio[1]-ratio[0], np.log(competition[1]/competition[0]))
        common = torch.tensor((ratio[0]+ratio[1])/2)[None]
        contrast = torch.tensor(ratio[1]-ratio[0])[None]
        torch.testing.assert_close(domain_logits((common,contrast),torch.ones(2)),torch.tensor(ratio[1])[None])
        torch.testing.assert_close(domain_logits((common,contrast),torch.zeros(2)),torch.tensor(ratio[0])[None])

    def test_source_labels_only_affect_ratio_objective(self):
        common = torch.randn(2,8, requires_grad=True)
        contrast = torch.randn(2,8, requires_grad=True)
        y = torch.tensor([0.,0.,1.,1.,0.,0.,1.,1.])
        changed = y.clone(); changed[4:] = 1-changed[4:]
        domain = torch.tensor([1.,1.,1.,1.,0.,0.,0.,0.])
        teacher = torch.randn(8)
        for arm in ['direct','teacher_offset']:
            a = loss_for_arm((common,contrast),y,domain,teacher,arm,.4)
            b = loss_for_arm((common,contrast),changed,domain,teacher,arm,.4)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        a = torch.autograd.grad(loss_for_arm((common,contrast),y,domain,teacher,'conditional_ratio',.4),contrast,retain_graph=True)[0]
        b = torch.autograd.grad(loss_for_arm((common,contrast),changed,domain,teacher,'conditional_ratio',.4),contrast)[0]
        self.assertGreater(float((a-b).abs().max()),0)

    def test_zero_contrast_fallback_and_label_free_forward(self):
        torch.manual_seed(4)
        model = RatioMLP({'numeric':['x'],'cat_sizes':[8]},dict(members=2,embedding_dim=2,frequencies=2,frequency_sigma=1.,hidden=[8],dropout=0))
        x = dict(num=torch.randn(7,1),missing=torch.zeros(7,1),cat=torch.zeros(7,1,dtype=torch.long))
        model.eval()
        out = model(x)
        teacher = torch.randn(7)
        torch.testing.assert_close(query_logits(out,teacher,'conditional_ratio',0),teacher[None].expand(2,-1))
        altered = dict(x, satisfaction=torch.ones(7))
        for a,b in zip(model(x),model(altered)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
