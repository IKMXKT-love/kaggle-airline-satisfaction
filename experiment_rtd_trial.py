"""One fixed RTD/no-RTD transfer comparison with a two-hour deadline."""
import sys,time,json,math,gc,hashlib
from pathlib import Path
from rtd_trial_support import ROOT,RUN,RowNet,encode_data,write,guard,corrupt,save_state,load_state,predict
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

P=json.loads((RUN/'protocol.json').read_text('utf-8'))
sys.stdout.reconfigure(encoding='utf-8')

def progress(**fields):
    value={'elapsed_seconds':time.time()-P['start_epoch'],**fields}
    write('progress.json',value);print(json.dumps(value),flush=True)

def create(state):
    torch.manual_seed(P['seed']);return RowNet(state).cuda()

def contracts(net,raw):
    net.eval();a=raw[:16].clone();b=raw[100:116].clone()
    gen=torch.Generator(device='cuda').manual_seed(888)
    mask=torch.rand(a.shape,device='cuda',generator=gen)<.15
    altered=a.clone();altered[mask]=b[mask]
    one=corrupt(a,b,mask);two=corrupt(altered,b,mask)
    assert torch.equal(one,two)
    with torch.no_grad():
        v1=net.views(one);v2=net.views(two)
        assert all(torch.equal(x,y) for x,y in zip(v1,v2))
        # Test set symmetry with explicit fp32 backbone and fixed CLS location.
        tok=net.tokenize(one[:2]);perm=torch.cat([torch.tensor([0],device='cuda'),torch.randperm(23,device='cuda')+1])
        pos=torch.zeros(tok.shape[:2],device='cuda',dtype=torch.long)
        h1=net.backbone(inputs_embeds=tok,position_ids=pos).last_hidden_state
        h2=net.backbone(inputs_embeds=tok[:,perm],position_ids=pos).last_hidden_state
        z1=net.class_head(h1[:,1:].mean(1));z2=net.class_head(h2[:,1:].mean(1))
        error=float((z1-z2).abs().max());assert error<5e-5,error
    write('contracts.json',{'raw_replacement_invariance':True,'all_derived_views_recomputed':True,'fp32_permutation_max_logit_error':error,'labels_absent_from_tokenizer':True})

def train(net,raw,indices,task,epochs,labels=None,stop=None):
    main=list(net.backbone.parameters())
    rest=[p for name,p in net.named_parameters() if not name.startswith('backbone.')]
    opt=torch.optim.AdamW([{'params':main,'lr':P['backbone_lr']},{'params':rest,'lr':P['tokenizer_lr']}],weight_decay=0,fused=True)
    steps=math.ceil(len(indices)/P['microbatch']);acc=P['effective_batch']//P['microbatch']
    total=epochs*math.ceil(steps/acc);update=0;best=-float('inf');history=[]
    generator=torch.Generator(device='cuda').manual_seed(P['seed']+101)
    best_round=None
    for epoch in range(epochs):
        order=indices[torch.randperm(len(indices),device='cuda',generator=generator)]
        net.train();opt.zero_grad(set_to_none=True);loss_sum=0.;epochstart=time.time()
        for step,begin in enumerate(range(0,len(order),P['microbatch'])):
            guard(reserve=180)
            idx=order[begin:begin+P['microbatch']];receiver=raw[idx]
            if task=='rtd':
                donor_indices=torch.randint(len(raw),(len(idx),21),device='cuda',generator=generator)
                donor=raw[donor_indices,torch.arange(21,device='cuda')[None,:]]
                target=(torch.rand(receiver.shape,device='cuda',generator=generator)<P['rtd_probability'])
                values=corrupt(receiver,donor,target);target=target.float()
            else:values=receiver;target=labels[idx]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net(values,task);loss=F.binary_cross_entropy_with_logits(logits.float(),target)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            (loss/acc).backward();loss_sum+=float(loss.detach())*len(idx)
            if (step+1)%acc==0 or step==steps-1:
                frac=update/max(total-1,1);warm=P['warmup_fraction']
                factor=min((frac+1/total)/warm,1.) if frac<warm else max((1-frac)/(1-warm),0.)
                opt.param_groups[0]['lr']=P['backbone_lr']*factor
                opt.param_groups[1]['lr']=P['tokenizer_lr']*factor
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient norm')
                opt.step();opt.zero_grad(set_to_none=True);update+=1
            if step%100==0:
                progress(stage=task,epoch=epoch+1,epochs=epochs,step=step+1,steps=steps,
                         epoch_rows_per_second=(begin+len(idx))/max(time.time()-epochstart,.01),mean_loss=loss_sum/(begin+len(idx)))
        item={'epoch':epoch+1,'mean_loss':loss_sum/len(indices),'seconds':time.time()-epochstart}
        if task=='class':
            z=predict(net,raw,stop,P['eval_batch']);ys=labels[stop].cpu().numpy()
            auc=roc_auc_score(ys,z);item['stopping_auc']=float(auc)
            if auc>best:
                best=auc;best_round=epoch+1;save_state(net,ACTIVE_ARM+'_best.pt')
        else:save_state(net,'pretrained.pt')
        history.append(item);write(ACTIVE_ARM+'_'+task+'_history.json',history)
        progress(stage=task,epoch_completed=epoch+1,metrics=item)
    del opt;gc.collect();torch.cuda.empty_cache()
    return history,best_round

def main():
    global ACTIVE_ARM
    guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    raw_np,state=encode_data();raw=torch.tensor(raw_np,device='cuda');del raw_np
    fold=np.load(ROOT/'research_s6e10/extended_review_20261020/reference_predictions/dariushafshar__folds_seed42.npy')
    fit_np=np.flatnonzero(np.isin(fold,P['fit_folds']));stop_np=np.flatnonzero(fold==4);score_np=np.flatnonzero(fold==0)
    assert len(fit_np)==419781 and not set(fit_np)&set(stop_np) and not set(fit_np)&set(score_np)
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype='float32')
    labels=torch.tensor(y,device='cuda')
    fit=torch.tensor(fit_np,device='cuda');stop=torch.tensor(stop_np,device='cuda');score=torch.tensor(score_np,device='cuda')
    np.savez(RUN/'indices.npz',fit=fit_np,stop=stop_np,score=score_np)
    net=create(state);contracts(net,raw)
    progress(stage='prepared',fit_rows=len(fit_np),stop_rows=len(stop_np),score_rows=len(score_np),pretrain_rows=len(raw))
    # Baseline first guarantees a complete no-RTD reference before the expensive arm.
    ACTIVE_ARM='clean';h,b=train(net,raw,fit,'class',P['classification_epochs'],labels,stop)
    load_state(net,'clean_best.pt')
    np.savez(RUN/'clean_predictions.npz',stop_logit=predict(net,raw,stop,P['eval_batch']),score_logit=predict(net,raw,score,P['eval_batch']))
    del net;gc.collect();torch.cuda.empty_cache()
    net=create(state);ACTIVE_ARM='rtd';preidx=torch.arange(len(raw),device='cuda')
    train(net,raw,preidx,'rtd',P['pretrain_epochs'])
    train(net,raw,fit,'class',P['classification_epochs'],labels,stop)
    load_state(net,'rtd_best.pt')
    np.savez(RUN/'rtd_predictions.npz',stop_logit=predict(net,raw,stop,P['eval_batch']),score_logit=predict(net,raw,score,P['eval_batch']))
    del net,raw,labels;gc.collect();torch.cuda.empty_cache()
    progress(stage='training_complete')

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('error.json',{'type':type(e).__name__,'message':str(e),'elapsed_seconds':time.time()-P['start_epoch']})
        raise
