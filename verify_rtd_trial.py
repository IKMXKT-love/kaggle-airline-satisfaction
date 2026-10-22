"""Rebuild all scored/stopping model outputs plus independent numpy input views."""
import json,time,hashlib,sys,gc
from pathlib import Path
from rtd_trial_support import ROOT,RUN,RowNet,encode_data,load_state,predict,write,guard,NUM
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    p=json.loads((RUN/'protocol.json').read_text('utf-8'))
    state=json.loads((RUN/'encoder.json').read_text('utf-8'))
    raw,newstate=encode_data();assert newstate==state
    ix=np.load(RUN/'indices.npz')
    # Independently calculate all four input views on a fixed 257-row sample.
    selected=np.random.default_rng(227).choice(len(raw),257,replace=False)
    a=raw[selected];ids=a.astype('int64');ids[:,2]=np.clip(ids[:,2],0,127);ids[:,5]=np.array(state['fd_lut'])[ids[:,5]]
    edges=np.array([0.,5.,15.,30.,60.,120.])
    ids[:,19:21]=np.searchsorted(edges,a[:,19:21],side='left');ids[:,19:21][a[:,19:21]<0]=7
    num=a[:,NUM].copy();num[:,1:]=np.log1p(np.maximum(num[:,1:],0));num=(num-np.array(state['numeric_mean'],dtype='float32'))/np.array(state['numeric_std'],dtype='float32')
    rat=a[:,6:19]/5
    proportions=np.column_stack([(a[:,6:19]==i).mean(1) for i in range(6)]).astype('float32')
    stats=np.column_stack([rat.mean(1),rat.std(1),rat.min(1),rat.max(1)])
    v28=np.column_stack([rat,proportions,stats,num,num[:,3]-num[:,2]])
    v17=np.column_stack([rat,num])
    gpu=torch.tensor(raw,device='cuda');out={}
    for arm in ['clean','rtd']:
        guard();net=RowNet(state).cuda();load_state(net,arm+'_best.pt');net.eval()
        with torch.no_grad():views=net.views(gpu[selected])
        errors=[float(np.max(np.abs(x.cpu().numpy()-v))) for x,v in zip(views,[ids,num,v28,v17])]
        assert errors[0]==0 and max(errors[1:])<2e-5,errors
        old=np.load(RUN/(arm+'_predictions.npz'))
        rebuilt={}
        for name in ['stop','score']:
            idx=torch.tensor(ix[name],device='cuda')
            z=predict(net,gpu,idx,p['eval_batch']);err=float(np.max(np.abs(z-old[name+'_logit'])))
            assert err<=1e-7,err
            rebuilt[name]={'rows':len(idx),'max_logit_error':err}
        out[arm]={'independent_view_errors':errors,'rebuilt':rebuilt}
        del net;gc.collect();torch.cuda.empty_cache()
    # Independent public-matrix/meta reconstruction for every outer score row.
    score=ix['score'];saved=np.load(RUN/'scored_meta_predictions.npz')
    matrix=np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/train_logits.npy',mmap_mode='r')
    x=np.array(matrix[score,:246]);rebuild={}
    for arm in ['matched_baseline','clean','rtd','incumbent']:
        if arm=='incumbent':w=np.load(ROOT/'.artifact_runtime/public_pool_trial_20261020/public_only_fold0.npz')['weights'];xx=x
        else:
            w=np.load(RUN/(arm+'_meta.npz'))['weights']
            xx=x if arm=='matched_baseline' else np.column_stack([x,np.load(RUN/(arm+'_predictions.npz'))['score_logit']])
        with threadpool_limits(limits=4):z=np.sum(xx*w[:-1],axis=1)+w[-1]
        error=float(np.max(np.abs(z-saved[arm])));assert error<1e-12,error
        rebuild[arm]=error
    preserved={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in p['preserved'].items()};assert all(preserved.values())
    write('verification.json',{'models':out,'all_meta_rows_rebuilt_max_error':rebuild,'preserved':preserved,'elapsed_seconds':time.time()-p['start_epoch'],'all_pass':True})
    print('VERIFICATION_PASSED',flush=True)

if __name__=='__main__':main()
