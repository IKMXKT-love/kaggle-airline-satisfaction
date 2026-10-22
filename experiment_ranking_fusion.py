"""One frozen uniform-pair/logloss fusion experiment; native meta folds only."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from fit_public_pool import newton, predict

ROOT = Path(__file__).resolve().parent
RUN = ROOT / '.artifact_runtime/ranking_fusion_20261022'
BASE = ROOT / '.artifact_runtime/public_pool_trial_20261020'
FOLDS = ROOT / 'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy'
sys.stdout.reconfigure(encoding='utf-8')

def write(name, value):
    (RUN/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for b in iter(lambda:stream.read(2**20),b''): h.update(b)
    return h.hexdigest()

def gpu_objective(a, y, pos, neg, reg, mix):
    def fun(w):
        wt=torch.as_tensor(w,device='cuda',dtype=torch.float64)
        z=a@wt[:-1]+wt[-1]
        p=z.sigmoid()
        value=(1-mix)*(torch.nn.functional.softplus(z)-y*z).mean()
        residual=(1-mix)*(p-y)/len(y)
        d=z[pos]-z[neg]
        value=value+mix*torch.nn.functional.softplus(-d).mean()+reg*wt[:-1].square().sum()/2
        pair_grad=-mix*torch.sigmoid(-d)/len(pos)
        residual.index_add_(0,pos,pair_grad)
        residual.index_add_(0,neg,-pair_grad)
        grad=torch.cat([a.T@residual+reg*wt[:-1],residual.sum().reshape(1)])
        return float(value.cpu()),grad.cpu().numpy()
    return fun

def check_objective():
    rng=np.random.default_rng(183)
    a=rng.normal(size=(41,7)); y=np.arange(41)%2
    pos=rng.choice(np.flatnonzero(y),200);neg=rng.choice(np.flatnonzero(1-y),200)
    w=rng.normal(size=8); reg=.03;mix=.5
    f=gpu_objective(torch.tensor(a,device='cuda'),torch.tensor(y,device='cuda',dtype=torch.float64),torch.tensor(pos,device='cuda'),torch.tensor(neg,device='cuda'),reg,mix)
    value,grad=f(w); z=a@w[:-1]+w[-1];d=(a[pos]-a[neg])@w[:-1]
    r=(1-mix)*(expit(z)-y)/len(y)
    g=np.r_[a.T@r+reg*w[:-1]+(a[pos]-a[neg]).T@(-mix*expit(-d)/len(pos)),r.sum()]
    v=(1-mix)*np.mean(np.logaddexp(0,z)-y*z)+mix*np.mean(np.logaddexp(0,-d))+reg*np.dot(w[:-1],w[:-1])/2
    assert abs(value-v)<1e-12 and np.max(abs(g-grad))<1e-12
    return {'value_error':abs(value-v),'gradient_error':float(np.max(abs(g-grad))),'reference':'CPU explicit pair differences'}

def main():
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    deadline=p['started_epoch']+p['budget_seconds']
    def check():
        if time.time()>deadline: raise TimeoutError('Cumulative900s budget')
    torch.set_num_threads(4)
    check();write('objective_check.json',check_objective())
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    folds=np.load(FOLDS,allow_pickle=False)
    x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    assert x.shape==(len(y),253)
    fingerprint={'train':sha(ROOT/'train.csv'),'folds':sha(FOLDS),'manifest':sha(BASE/'matrix_manifest.json'),'train_logits':sha(BASE/'train_logits.npy')}
    write('inputs.json',fingerprint)
    results=[]
    def stage(tag,train,score,baseline_path=None):
        check();started=time.time()
        assert not np.any(train&score)
        if baseline_path is None:
            with threadpool_limits(limits=4):
                baseline,bmeta=newton(x,y,train,246,deadline)
        else:
            baseline=np.load(baseline_path,allow_pickle=False)['weights'].copy();bmeta={'reused':True}
        np.savez(RUN/f'{tag}_baseline.npz',weights=baseline)
        local_y=y[train];a=torch.tensor(np.array(x[train,:246]),device='cuda',dtype=torch.float64)
        yt=torch.tensor(local_y,device='cuda',dtype=torch.float64)
        rng=np.random.default_rng(p['seed'])
        pi=rng.choice(np.flatnonzero(local_y==1),p['pair_count'])
        ni=rng.choice(np.flatnonzero(local_y==0),p['pair_count'])
        pt=torch.tensor(pi,device='cuda');nt=torch.tensor(ni,device='cuda')
        objective=gpu_objective(a,yt,pt,nt,1/(.003*len(local_y)),.5)
        calls=0
        def fun(w):
            nonlocal calls
            check();calls+=1
            if calls%40==0: print(tag,'evaluations',calls,'seconds',round(time.time()-started,1),flush=True)
            return objective(w)
        opt=minimize(fun,baseline,jac=True,method='L-BFGS-B',options={'maxiter':p['max_iterations'],'maxcor':30,'gtol':1e-7,'ftol':1e-12,'maxls':30})
        value,gradient=objective(opt.x)
        converged=bool(opt.success and np.max(abs(gradient))<2e-5)
        np.savez(RUN/f'{tag}_candidate.npz',weights=opt.x)
        meta={'success':bool(opt.success),'converged':converged,'iterations':int(opt.nit),'calls':calls,'message':str(opt.message),'gradient_max':float(np.max(abs(gradient))),'objective':value,'training_rows':int(train.sum()),'fitting_seconds':time.time()-started,'baseline_fit':bmeta}
        write(f'{tag}_fit.json',meta)
        del a,yt,pt,nt,objective;torch.cuda.empty_cache()
        if not converged: raise RuntimeError('Candidate did not converge; no scoring/retuning')
        idx=np.flatnonzero(score)
        with threadpool_limits(limits=4):
            b=x[idx,:246]@baseline[:-1]+baseline[-1]
            c=x[idx,:246]@opt.x[:-1]+opt.x[-1]
        np.savez(RUN/f'{tag}_scored.npz',index=idx,baseline=b,candidate=c)
        ba=float(roc_auc_score(y[idx],b));ca=float(roc_auc_score(y[idx],c))
        result={'tag':tag,'rows':len(idx),'baseline_auc':ba,'candidate_auc':ca,'delta':ca-ba,'seconds':time.time()-started}
        results.append(result);write('results.json',results);print('RESULT',json.dumps(result),flush=True)
        return result
    inner=stage('inner',np.isin(folds,[1,2,3]),folds==4)
    if inner['delta']<p['inner_gate_delta']:
        status='stopped_inner_gate'
    else:
        first=stage('fold0',folds!=0,folds==0,BASE/'public_only_fold0.npz')
        if first['delta']<p['outer_screen_gate_delta']: status='stopped_outer_screen_gate'
        else:
            estimate=first['seconds']*4+60
            if time.time()+estimate>deadline: status='stopped_budget_projection'
            else:
                for f in range(1,5): stage(f'fold{f}',folds!=f,folds==f,BASE/f'public_only_fold{f}.npz')
                ds=np.array([r['delta'] for r in results if r['tag'].startswith('fold')])
                passed=ds.mean()>=p['confirmation_gate_mean_delta'] and (ds>0).sum()>=4 and ds.mean()-2*ds.std(ddof=1)/np.sqrt(5)>0
                status='confirmation_passed' if passed else 'stopped_confirmation_gate'
    write('completion.json',{'status':status,'elapsed_seconds':time.time()-p['started_epoch'],'results':results,'submission_generated':False})
    print('COMPLETE',status,flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        write('failure.json',{'type':type(e).__name__,'error':str(e),'at_epoch':time.time()});raise
