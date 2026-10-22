"""Local RTD transfer experiment. All views derive from the corrupted raw row."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'.artifact_runtime/tabicl_finetune_deps'),str(ROOT/'.artifact_runtime/tabicl_runtime')]
import json,math,time,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import ModernBertConfig,ModernBertModel

RUN=ROOT/'.artifact_runtime/rtd_trial_20261022'
RATINGS=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
COLS=['Gender','Customer Type','Age','Type of Travel','Class','Flight Distance',*RATINGS,'Departure Delay in Minutes','Arrival Delay in Minutes']
CAT=[0,1,3,4]
NUM=[2,5,19,20]

def write(name,obj):
    path=RUN/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def guard(reserve=0):
    p=json.loads((RUN/'protocol.json').read_text('utf-8'))
    if time.time()>p['deadline_epoch']-reserve:raise TimeoutError('Two-hour cumulative deadline reached')

def encode_data():
    frames=[pd.read_csv(ROOT/p,usecols=COLS)[COLS] for p in ['train.csv','test.csv','external_data/train.csv','external_data/test.csv']]
    frame=pd.concat(frames,ignore_index=True)
    vocab={}
    for j in CAT:
        c=COLS[j];v=sorted(frame[c].dropna().unique().tolist());vocab[c]=v
        frame[c]=frame[c].map({s:i for i,s in enumerate(v)}).fillna(len(v))
    raw=frame.to_numpy(dtype='float32',copy=True)
    raw[~np.isfinite(raw)]=-1
    assert np.all((raw[:,6:19]>=0)&(raw[:,6:19]<=5))
    fd=raw[:,5].astype('int64');counts=np.bincount(fd)
    lut=np.zeros(len(counts),dtype='int64');keep=np.flatnonzero(counts>=3);lut[keep]=np.arange(1,len(keep)+1)
    numbers=raw[:,NUM].copy();numbers[:,1:]=np.log1p(np.maximum(numbers[:,1:],0))
    mean=numbers.mean(0,dtype='float64');std=np.maximum(numbers.std(0,dtype='float64'),.01)
    state={'columns':COLS,'vocab':vocab,'fd_lut':lut.tolist(),'numeric_mean':mean.tolist(),'numeric_std':std.tolist(),
           'row_counts':[len(x) for x in frames],'feature_sha256':hashlib.sha256(raw.tobytes()).hexdigest()}
    write('encoder.json',state)
    return raw,state

class RowNet(nn.Module):
    def __init__(self,state):
        super().__init__()
        cfg=ModernBertConfig.from_dict(json.loads((ROOT/'.artifact_runtime/rtd_preflight_20261022/upstream_config.json').read_text('utf-8')))
        cfg.reference_compile=False;cfg._attn_implementation='sdpa';cfg._attn_implementation_internal='sdpa'
        self.backbone=ModernBertModel(cfg)
        sizes=[len(state['vocab'][COLS[j]])+1 if j in CAT else (int(max(state['fd_lut']))+1 if j==5 else (128 if j==2 else (6 if 6<=j<=18 else 8))) for j in range(21)]
        offsets=np.cumsum([0]+sizes[:-1]).tolist()
        self.value=nn.Embedding(sum(sizes),768);nn.init.normal_(self.value.weight,std=.02)
        self.field=nn.Parameter(torch.randn(21,768)*.02)
        self.frequencies=nn.Parameter(torch.randn(4,32))
        self.numeric=nn.ModuleList([nn.Linear(64,768) for _ in NUM])
        self.row_summary=nn.Linear(28,768);self.numeric_summary=nn.Linear(17,768)
        self.cls=nn.Parameter(torch.randn(1,1,768)*.02)
        self.rtd_head=nn.Linear(768,1);self.class_head=nn.Linear(768,1)
        self.register_buffer('offsets',torch.tensor(offsets));self.register_buffer('fd_lut',torch.tensor(state['fd_lut']))
        self.register_buffer('num_mean',torch.tensor(state['numeric_mean'],dtype=torch.float32));self.register_buffer('num_std',torch.tensor(state['numeric_std'],dtype=torch.float32))
        self.register_buffer('delay_edges',torch.tensor([0.,5.,15.,30.,60.,120.]))

    def views(self,raw):
        ids=raw.long().clone();ids[:,2]=ids[:,2].clamp(0,127);ids[:,5]=self.fd_lut[ids[:,5]]
        ids[:,19:21]=torch.bucketize(raw[:,19:21].contiguous(),self.delay_edges,right=False)
        ids[:,19:21]=torch.where(raw[:,19:21]<0,7,ids[:,19:21])
        num=raw[:,NUM].clone();num[:,1:]=torch.log1p(num[:,1:].clamp_min(0));num=(num-self.num_mean)/self.num_std
        ratings=raw[:,6:19]/5
        proportions=torch.stack([(raw[:,6:19]==i).float().mean(1) for i in range(6)],dim=1)
        stats=torch.stack([ratings.mean(1),ratings.std(1,unbiased=False),ratings.min(1).values,ratings.max(1).values],dim=1)
        gap=(num[:,3]-num[:,2])[:,None]
        s28=torch.cat([ratings,proportions,stats,num,gap],dim=1)
        s17=torch.cat([ratings,num],dim=1)
        return ids,num,s28,s17

    def tokenize(self,raw):
        ids,num,s28,s17=self.views(raw)
        tokens=self.value(ids+self.offsets)+self.field
        extra=torch.zeros_like(tokens)
        for k,j in enumerate(NUM):
            phase=2*math.pi*num[:,k,None]*self.frequencies[k]
            extra[:,j]=self.numeric[k](torch.cat([phase.sin(),phase.cos()],dim=1))
        tokens=tokens+extra
        tokens=torch.nn.functional.rms_norm(tokens,(768,))*.03
        summaries=torch.stack([self.row_summary(s28),self.numeric_summary(s17)],dim=1)
        summaries=torch.nn.functional.rms_norm(summaries,(768,))*.03
        return torch.cat([self.cls.expand(len(raw),-1,-1),tokens,summaries],dim=1)

    def forward(self,raw,task='class'):
        tokens=self.tokenize(raw)
        h=self.backbone(inputs_embeds=tokens,position_ids=torch.zeros(tokens.shape[:2],device=raw.device,dtype=torch.long)).last_hidden_state
        return self.rtd_head(h[:,1:22]).squeeze(-1) if task=='rtd' else self.class_head(h[:,1:].mean(1)).squeeze(-1)

def corrupt(receiver,donor,mask):
    return torch.where(mask,donor,receiver)

def save_state(net,name):
    # Unused language vocabulary is random and never enters inputs_embeds forward.
    state={k:v.detach().cpu() for k,v in net.state_dict().items() if k!='backbone.embeddings.tok_embeddings.weight'}
    path=RUN/name;tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def load_state(net,name):
    out=net.load_state_dict(torch.load(RUN/name,map_location='cpu',weights_only=True),strict=False)
    assert out.missing_keys==['backbone.embeddings.tok_embeddings.weight'] and not out.unexpected_keys,out

@torch.no_grad()
def predict(net,raw,index,batch=256):
    net.eval();result=[]
    for begin in range(0,len(index),batch):
        guard()
        with torch.autocast('cuda',dtype=torch.bfloat16):z=net(raw[index[begin:begin+batch]],'class')
        result.append(z.float().cpu().numpy())
    return np.concatenate(result).astype('float64')
