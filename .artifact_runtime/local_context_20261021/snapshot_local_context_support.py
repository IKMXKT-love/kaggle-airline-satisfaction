"""Fold-local retrieval and explicit context/query tasks for pinned TabICL."""
import copy

import numpy as np
import torch

from tabicl_finetune_support import SequentialViewsClassifier, inference_parameters
from tabicl._finetune.data import MetaBatch, _build_ensemble_generator

CATEGORIES = ['Gender', 'Customer Type', 'Type of Travel', 'Class']


class RetrievalEncoder:
    """Target-free robust numerics and one-hot categories; fit on reference rows."""
    def fit(self, frame):
        assert not {'id','satisfaction'} & set(frame.columns)
        self.columns = list(frame)
        self.numeric = [c for c in frame if c not in CATEGORIES]
        self.categories = {c:sorted(frame[c].astype(str).unique().tolist()) for c in CATEGORIES}
        a = self.numbers(frame)
        self.median = np.nanmedian(a,axis=0)
        quartiles = np.nanquantile(a,[.25,.75],axis=0)
        self.scale = np.maximum(quartiles[1]-quartiles[0],1.)
        assert np.isfinite(self.median).all() and np.isfinite(self.scale).all()
        return self

    def numbers(self, frame):
        a = frame[self.numeric].to_numpy(dtype='float64',copy=True)
        for j,c in enumerate(self.numeric):
            if c in ['Departure Delay in Minutes','Arrival Delay in Minutes']:
                a[:,j] = np.log1p(np.maximum(a[:,j],0))
        return a

    def transform(self, frame):
        assert list(frame) == self.columns
        a = self.numbers(frame)
        a = np.where(np.isfinite(a),a,self.median)
        a = np.clip((a-self.median)/self.scale,-4,4)
        parts = [a]
        for c in CATEGORIES:
            value = frame[c].astype(str).to_numpy()
            parts.append(np.column_stack([value==category for category in self.categories[c]]).astype('float32'))
        return np.concatenate(parts,axis=1).astype('float32')

    def state(self):
        return dict(columns=self.columns,numeric=self.numeric,categories=self.categories,
                    median=self.median.tolist(),scale=self.scale.tolist())


def explicit_meta_batch(x, y, context, query, seed):
    """Only context labels enter the forward inputs and fitted preprocessing.

    Uses the official ensemble transformation but accepts an already isolated
    split, ensuring random/local arms can share exactly the same query rows.
    """
    context, query = np.asarray(context),np.asarray(query)
    assert len(np.unique(context))==len(context) and len(np.unique(query))==len(query)
    assert not np.intersect1d(context,query).size
    assert context.min()>=0 and query.min()>=0 and max(context.max(),query.max())<len(x)
    assert set(y[context])=={0,1}
    generator = _build_ensemble_generator(classification=True,n_estimators=2,
        norm_methods=['none','power'],feat_shuffle_method='latin',class_shuffle_method='shift',
        outlier_threshold=4.,random_state=seed)
    generator.fit(x[context],y[context])
    transformed = generator.transform(x[query],mode='both')
    xs,ys,yq = [],[],[]
    for norm,(values,targets) in transformed.items():
        xs.append(values);ys.append(targets)
        for _,pattern in generator.ensemble_configs_[norm]:
            yq.append(np.asarray(pattern)[y[query].astype(int)] if pattern is not None else y[query])
    return MetaBatch(X=torch.from_numpy(np.concatenate(xs)).float(),
                     y_train=torch.from_numpy(np.concatenate(ys)).float(),
                     y_query=torch.from_numpy(np.stack(yq)).long(),train_size=len(context))


class TaskSampler:
    """Queries follow reference cluster mass; only reference indices are exposed."""
    def __init__(self, assignment, pools, context_size, query_size, background=128, seed=20261022):
        self.assignment,self.pools = np.asarray(assignment),np.asarray(pools)
        self.context_size,self.query_size,self.background,self.seed = context_size,query_size,background,seed
        self.n = len(assignment)
        self.groups = [np.flatnonzero(assignment==k) for k in range(len(pools))]
        self.weights = np.array([len(v) for v in self.groups],dtype=float)/self.n
        assert np.isclose(self.weights.sum(),1.)
        assert self.pools.min()>=0 and self.pools.max()<self.n

    def sample(self, step):
        rng = np.random.default_rng(self.seed+step*104729)
        group = int(rng.choice(len(self.groups),p=self.weights))
        q = rng.choice(self.groups[group],size=min(self.query_size,len(self.groups[group])),replace=False)
        available = self.pools[group][~np.isin(self.pools[group],q)]
        local = rng.choice(available,size=self.context_size-self.background,replace=False)
        excluded = np.r_[q,local]
        global_available = np.setdiff1d(np.arange(self.n),excluded,assume_unique=False)
        background = rng.choice(global_available,size=self.background,replace=False)
        random_available = np.setdiff1d(np.arange(self.n),np.r_[q,background],assume_unique=False)
        random_context = np.r_[rng.choice(random_available,size=self.context_size-self.background,replace=False),background]
        local_context = np.r_[local,background]
        rng.shuffle(random_context);rng.shuffle(local_context)
        for c in [random_context,local_context]:
            assert len(c)==self.context_size and len(np.unique(c))==len(c)
            assert not np.intersect1d(c,q).size
        return dict(anchor=group,query=q,random=random_context,local=local_context)


def inference_contexts(assignment,pools,context_size,background=128,seed=90210):
    result = {'random':[],'local':[]}
    n = len(assignment)
    for k,pool in enumerate(pools):
        rng = np.random.default_rng(seed+k)
        local = rng.choice(pool,context_size-background,replace=False)
        bg = rng.choice(np.setdiff1d(np.arange(n),local),background,replace=False)
        random = np.r_[rng.choice(np.setdiff1d(np.arange(n),bg),context_size-background,replace=False),bg]
        local = np.r_[local,bg]
        rng.shuffle(local);rng.shuffle(random)
        result['random'].append(random);result['local'].append(local)
    return {k:np.stack(v) for k,v in result.items()}


class ReusedNetworkClassifier(SequentialViewsClassifier):
    """Reuse one already loaded network; leave official fit/transforms intact."""
    def _load_model(self):
        self.model_ = self.shared_network
        self.model_.eval()
        self.model_config_ = copy.deepcopy(self.shared_config)
        self.model_path_ = self.model_path


def predict_context(net,config,checkpoint,x,y,context,query,query_batch=1024):
    assert set(y[context])=={0,1}
    estimator = ReusedNetworkClassifier(**inference_parameters(checkpoint,2))
    estimator.shared_network,estimator.shared_config = net,config
    try:
        estimator.fit(x[context],y[context])
        return estimator.predict_all(query,query_batch=query_batch,maximum_seconds=90)[:,1]
    finally:
        estimator.model_kv_cache_ = None
        net.clear_cache()
