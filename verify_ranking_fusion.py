"""CPU rebuild of every scored prediction and paired AUC differences."""
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiment_ranking_fusion import ROOT,RUN,BASE,FOLDS,sha,write
from multitask_support import paired

def main():
    protocol=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    results=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    x=np.load(BASE/'train_logits.npy',mmap_mode='r',allow_pickle=False)
    y=pd.read_csv(ROOT/'train.csv',usecols=['satisfaction']).satisfaction.to_numpy(dtype='int64')
    folds=np.load(FOLDS,allow_pickle=False)
    expected=np.empty(len(y),dtype=int)
    for f,(_,idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y)): expected[idx]=f
    np.testing.assert_array_equal(folds,expected)
    inputs=json.loads((RUN/'inputs.json').read_text(encoding='utf-8'))
    for name,path in [('train',ROOT/'train.csv'),('folds',FOLDS),('manifest',BASE/'matrix_manifest.json'),('train_logits',BASE/'train_logits.npy')]: assert sha(path)==inputs[name]
    out=[]
    for r in results:
        tag=r['tag'];fold=4 if tag=='inner' else int(tag[-1])
        idx=np.flatnonzero(folds==fold)
        z=np.load(RUN/f'{tag}_scored.npz',allow_pickle=False)
        np.testing.assert_array_equal(idx,z['index'])
        rebuilt={};errors={}
        for arm in ['baseline','candidate']:
            w=np.load(RUN/f'{tag}_{arm}.npz',allow_pickle=False)['weights']
            p=np.empty(len(idx))
            with threadpool_limits(limits=4):
                for start in range(0,len(idx),4096):
                    rows=idx[start:start+4096];p[start:start+len(rows)]=x[rows,:246]@w[:-1]+w[-1]
            errors[arm]=float(np.max(abs(p-z[arm])));assert errors[arm]<1e-10
            assert abs(roc_auc_score(y[idx],p)-r[arm+'_auc'])<1e-12
            rebuilt[arm]=p
        diff=paired(y[idx],rebuilt['candidate'],rebuilt['baseline'])
        assert abs(diff['delta']-r['delta'])<1e-12
        out.append({**r,'rebuild_max_errors':errors,'paired':diff})
    for name,digest in protocol['protected'].items(): assert sha(ROOT/name)==digest,name
    v={'verified':True,'native_folds_match_sklearn':True,'protected_files_unchanged':True,'inputs_sha256_match':True,'rows':out,'elapsed_seconds_including_verification':time.time()-protocol['started_epoch'],'inference_scope':'All scored rows reconstructed on CPU. Cached base predictions not retrained.','validation_scope':'Conditional meta-validation; historical and upstream dependencies remain.'}
    write('verification.json',v)
    completion=json.loads((RUN/'completion.json').read_text(encoding='utf-8'))
    lines=['# 排序目标融合：限时初筛结果','',f"结束状态：{completion['status']}。累计至核验 {v['elapsed_seconds_including_verification']:.2f} 秒。",'', '唯一固定候选：50%点式BCE + 50%均匀正负配对logistic损失，相同L2强度；没有追加调参或挑选融合权重。','', '| 分区 | 对照AUC | 候选AUC | 差值 | 配对95%区间 |','|---|---:|---:|---:|---|']
    for r in out: lines.append(f"|{r['tag']}|{r['baseline_auc']:.10f}|{r['candidate_auc']:.10f}|{r['delta']:+.10f}|{r['paired']['interval95']}|")
    lines+=['','区间是同一批固定预测的近似配对区间，不覆盖训练不确定性、历史尝试选择或上游OOF泄漏。内部评分由原生fold4承担，原生fold1/2/3拟合；若内部失败则没有进入原生五折复核。','','独立核验：全部计分预测从保存权重分块CPU重建，AUC重算；原生折与StratifiedKFold(5,shuffle=True,random_state=42)一致；输入指纹与最佳四文件哈希一致。公开基础模型没有重新训练，故不宣称端到端独立验证。','','未生成测试预测或提交文件；当前最佳仍为用户反馈0.96187。这个实验不能否定所有排序目标，只否决该固定配方继续投入。']
    (ROOT/'RANKING_FUSION_REPORT_20261022.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(v,indent=2))

if __name__=='__main__': main()
