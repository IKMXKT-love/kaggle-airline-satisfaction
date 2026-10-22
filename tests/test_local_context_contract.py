"""Query-label exclusion and matched-task checks for local-context training."""
import copy
import unittest

import numpy as np

from local_context_support import TaskSampler, explicit_meta_batch, RetrievalEncoder


class LocalContract(unittest.TestCase):
    def test_matched_queries_and_disjoint_reference_only_contexts(self):
        assignment = np.repeat(np.arange(4),128)
        pools = np.stack([np.arange(k*128,(k+1)*128) for k in range(4)])
        sampler = TaskSampler(assignment,pools,context_size=64,query_size=32,background=8)
        for step in range(12):
            a,b = sampler.sample(step),sampler.sample(step)
            np.testing.assert_array_equal(a['query'],b['query'])
            for kind in ['random','local']:
                self.assertEqual(len(a[kind]),64)
                self.assertEqual(len(np.unique(a[kind])),64)
                self.assertFalse(np.intersect1d(a[kind],a['query']).size)
                self.assertGreaterEqual(a[kind].min(),0)
                self.assertLess(a[kind].max(),len(assignment))

    def test_query_labels_do_not_change_forward_input_or_context_preprocessing(self):
        rng = np.random.default_rng(9)
        x,y = rng.normal(size=(100,5)),np.tile([0,1],50)
        context,query = np.arange(64),np.arange(64,96)
        altered = y.copy();altered[query] = 1-altered[query]
        a,b = explicit_meta_batch(x,y,context,query,19),explicit_meta_batch(x,altered,context,query,19)
        np.testing.assert_array_equal(a.X,b.X)
        np.testing.assert_array_equal(a.y_train,b.y_train)
        self.assertFalse(np.array_equal(a.y_query,b.y_query))
        with self.assertRaises(AssertionError): explicit_meta_batch(x,y,context,np.arange(60,96),19)

    def test_retrieval_transform_does_not_refit_on_heldout_values(self):
        import pandas as pd
        fit = pd.DataFrame({'Age':[10,20,30,40],'Arrival Delay in Minutes':[0.,1.,2.,3.],
            'Gender':['a','b','a','b'],'Customer Type':['a']*4,'Type of Travel':['a']*4,'Class':['a']*4})
        encoder = RetrievalEncoder().fit(fit)
        before = copy.deepcopy(encoder.state())
        held = fit.iloc[:1].copy();held['Gender']='unseen';held['Age']=99999
        values = encoder.transform(held)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(before,encoder.state())
        with self.assertRaises(AssertionError): RetrievalEncoder().fit(fit.assign(satisfaction=0))


if __name__=='__main__': unittest.main()
