"""Finite feature ablations and complementary-model experiments; preserve v3."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from v2_training import ROOT, load_data, write_json
from v3_features import auxiliary, frames
from v3_training import fit_model

OUT = ROOT / '.artifact_runtime/v4'
GROUPS = {
    'route': lambda c: c.startswith('route_mean__'),
    'frequency': lambda c: c.startswith('frequency__'),
    'target_encoding': lambda c: c.startswith('te__'),
    'teacher': lambda c: c == 'teacher_logit',
    'auxiliary': lambda c: c.startswith('aux_expected_rating_'),
}
SPECS = ([dict(name='xgb_control', family='xgb', transform='full')] +
         [dict(name='without_' + g, family='xgb', transform='without_' + g) for g in GROUPS] +
         [dict(name='xgb_residual', family='xgb', transform='residual')] +
         [dict(name='lgb_' + str(n), family='lightgbm', transform='full', leaves=n)
          for n in [15, 63, 127]])
SIGNATURE_FILES = ['optimize_v4.py', 'v3_features.py', 'v3_training.py',
                   'v2_training.py', 'optimized_features.py', 'optimized_training.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'preserved.json'
    if not manifest.exists():
        names = ['main.py', 'train.csv', 'test.csv', 'sample_submission.csv',
                 'submission_optimized.csv', 'submission_v3.csv', 'submission_v3_oof.csv',
                 'v3_recipe.json', 'v3_confirmation.json', 'v3_verification.json',
                 'submission_v3_metrics.json', 'leaderboard_feedback.json']
        write_json(manifest, {n: sha(ROOT/n) for n in names})
    for n, h in read(manifest).items():
        if sha(ROOT/n) != h:
            raise ValueError('Protected file changed: ' + n)
    signature = hashlib.sha256(json.dumps({p: sha(ROOT/p) for p in SIGNATURE_FILES},
                                          sort_keys=True).encode() + manifest.read_bytes()).hexdigest()
    directory = OUT / signature[:20]
    directory.mkdir(exist_ok=True)
    write_json(directory/'protocol.json', dict(signature=signature, candidates=SPECS,
        split='existing fit/selection; development/audit for frozen confirmation',
        ablation='same XGBoost parameters and seed, each with own early stopping',
        mixture_weights=[.1, .2, .3, .4], replacement_weight=.4,
        gate_selection_gain=.00002, gate_confirmation_gain=.00002,
        confirmation_already_seen=True, no_audit_weight_tuning=True,
        additional_external_predictions=False, target='satisfaction', excluded=['id', 'satisfaction']))
    return directory


def transform(parts, mode):
    if mode.startswith('without_'):
        group = mode[len('without_'):]
        return tuple(p.drop(columns=[c for c in p if GROUPS[group](c)])
                     if p is not None else None for p in parts)
    if mode == 'residual':
        ratings = read(ROOT/'.artifact_runtime/v3/auxiliary.json')['ratings']
        result = []
        for p in parts:
            if p is None:
                result.append(None)
                continue
            p = p.copy()
            for i, col in enumerate(ratings):
                p['rating_residual_' + str(i)] = p[col] - p['aux_expected_rating_' + str(i)]
            result.append(p)
        return tuple(result)
    if mode != 'full':
        raise ValueError(mode)
    return parts


def train(spec, a, ya, b, yb, c=None, seed=42, rounds=None):
    a, b, c = transform((a, b, c), spec['transform'])
    assert not {'id', 'satisfaction'}.intersection(a.columns)
    assert list(a.columns) == list(b.columns)
    if spec['family'] == 'xgb':
        pv, pt, details = fit_model(a, ya, b, yb, c, family='xgb', seed=seed,
                                    device='cuda', rounds=rounds)
    else:
        import lightgbm as lgb
        begin = time.time()
        model = lgb.LGBMClassifier(n_estimators=rounds or 4000, learning_rate=.025,
            num_leaves=spec['leaves'], max_depth=-1, min_child_samples=100,
            colsample_bytree=.85, subsample=.9, subsample_freq=1,
            reg_lambda=10, reg_alpha=.1, max_bin=255, verbosity=-1,
            n_jobs=4, random_state=seed, deterministic=True, force_col_wise=True)
        kwargs = {} if rounds else dict(eval_set=[(b, yb)], eval_metric='auc',
            callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
        model.fit(a, ya, **kwargs)
        pv = model.predict_proba(b)[:, 1]
        pt = model.predict_proba(c)[:, 1] if c is not None else None
        details = dict(family='lightgbm', seed=seed, rounds=int(rounds or model.best_iteration_),
            auc=float(roc_auc_score(yb, pv)), train_auc=float(roc_auc_score(ya, model.predict_proba(a)[:, 1])),
            features=list(a.columns), seconds=time.time()-begin, version=lgb.__version__)
        del model
    assert np.isfinite(pv).all() and ((pv >= 0) & (pv <= 1)).all()
    details['spec'] = spec
    gc.collect()
    return pv, pt, details


def data():
    x, xt, y, ox, oy, prior, pt, stats, meta = load_data()
    aux, auxt = auxiliary(x, xt, ox)
    return x, xt, y, prior, pt, aux, auxt


def screen(directory):
    x, xt, y, prior, pt, aux, auxt = data()
    with np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz') as s:
        fit, valid = s['fit'], s['selection']
    assert not np.intersect1d(fit, valid).size
    a, b = frames(x, None, y, fit, valid, prior, pt, aux, auxt)
    for spec in SPECS:
        path = directory/(spec['name'] + '_selection.npz')
        if path.exists() and path.with_suffix('.json').exists():
            continue
        print('SCREEN START', spec['name'], flush=True)
        pv, _, d = train(spec, a, y[fit], b, y[valid])
        np.savez_compressed(path, valid_index=valid, prediction=pv)
        write_json(path.with_suffix('.json'), d)
        print('SCREEN RESULT', spec['name'], d['auc'], 'rounds', d['rounds'], flush=True)
    old = ROOT/'.artifact_runtime/writeup_search'
    v3 = ROOT/'.artifact_runtime/v3'
    bx = np.load(v3/'xgb_selection.npy')
    base = (.4*(.25*np.load(old/'te_original_d7_selection.npy') +
                .75*np.load(old/'catboost_allcat_original_d8_selection.npy')) +
            .4*bx + .2*np.load(v3/'realmlp_selection.npy'))
    baseline_auc = float(roc_auc_score(y[valid], base))
    control_auc = read(directory/'xgb_control_selection.json')['auc']
    trials, ablations = [], []
    for spec in SPECS:
        path = directory/(spec['name'] + '_selection.npz')
        with np.load(path) as z:
            np.testing.assert_array_equal(z['valid_index'], valid)
            pv = z['prediction']
        d = read(path.with_suffix('.json'))
        if spec['name'].startswith('without_'):
            ablations.append(dict(group=spec['name'][8:], auc=d['auc'],
                                  change_without_group=d['auc']-control_auc))
        if spec['name'] == 'xgb_control':
            continue
        choices = [('replace_xgb', .4)] if spec['family'] == 'xgb' else [('mix', w) for w in [.1,.2,.3,.4]]
        for mode, weight in choices:
            p = base + weight*(pv-bx) if mode == 'replace_xgb' else (1-weight)*base+weight*pv
            score = float(roc_auc_score(y[valid], p))
            trials.append(dict(spec=spec, mode=mode, weight=weight, auc=score,
                               gain=score-baseline_auc, rounds=d['rounds']))
    best = max(trials, key=lambda t:t['auc'])
    recipe = dict(**best, signature=directory.name, baseline_selection_auc=baseline_auc,
        selection_gain=best['gain'], ablations=ablations, trials=trials,
        control_auc=control_auc, prior_xgb_auc=float(roc_auc_score(y[valid], bx)),
        frozen_before_confirmation=True, audit_previously_seen=True,
        selection_minimum_gain=.00002, confirmation_minimum_gain=.00002)
    write_json(directory/'recipe.json', recipe)
    write_json(ROOT/'v4_recipe.json', recipe)
    print('FROZEN', best['spec']['name'], best['mode'], best['weight'], best['gain'], flush=True)
    return recipe


def combine(base, candidate, recipe, old_xgb=None):
    w = recipe['weight']
    if recipe['mode'] == 'replace_xgb':
        assert old_xgb is not None
        return base+w*(candidate-old_xgb)
    return (1-w)*base+w*candidate


def confirm(directory, recipe):
    path = directory/'confirmation.npz'
    if path.exists() and path.with_suffix('.json').exists():
        return read(path.with_suffix('.json'))
    x, xt, y, prior, pt, aux, auxt = data()
    with np.load(ROOT/'.artifact_runtime/writeup_search/split_indices.npz') as s:
        fit, valid = s['development'], s['audit']
    assert not np.intersect1d(fit, valid).size
    a, b = frames(x, None, y, fit, valid, prior, pt, aux, auxt)
    rounds = int(np.ceil(recipe['rounds']*1.1))
    print('CONFIRM START', recipe['spec']['name'], rounds, flush=True)
    pv, _, details = train(recipe['spec'], a, y[fit], b, y[valid], rounds=rounds)
    old_dir = Path(read(ROOT/'v3_confirmation.json')['directory'])
    base = np.load(old_dir/'blend.npy')
    p = combine(base, pv, recipe, np.load(old_dir/'xgb.npy'))
    auc, bs = map(float, (roc_auc_score(y[valid],p), roc_auc_score(y[valid],base)))
    result = dict(auc=auc, baseline_auc=bs, gain=auc-bs, labels_used_for_early_stopping=False,
        audit_previously_seen=True, details=details,
        passed=bool(recipe['selection_gain']>=.00002 and auc-bs>=.00002))
    np.savez_compressed(path, valid_index=valid, candidate=pv, baseline=base, prediction=p)
    write_json(path.with_suffix('.json'), result)
    write_json(ROOT/'v4_confirmation.json', result)
    print('CONFIRM RESULT', auc, 'gain', auc-bs, 'passed', result['passed'], flush=True)
    return result


def cv(directory, recipe):
    x, xt, y, prior, pt, aux, auxt = data()
    old = pd.read_csv(ROOT/'submission_v3_oof.csv')
    old_test = pd.read_csv(ROOT/'submission_v3.csv')
    old_dir = Path(read(ROOT/'submission_v3_metrics.json')['directory'])
    oof = np.zeros(len(y)); prediction = np.zeros(len(xt)); coverage = np.zeros(len(y), int)
    scores, baseline, details = [], [], []
    for fold, (fit, valid) in enumerate(StratifiedKFold(10,shuffle=True,random_state=42).split(x,y),1):
        np.testing.assert_array_equal(valid, np.flatnonzero(old.fold.to_numpy()==fold))
        path = directory/f'fold_{fold:02d}.npz'
        if not path.exists() or not path.with_suffix('.json').exists():
            a,b,c = frames(x,xt,y,fit,valid,prior,pt,aux,auxt)
            print('CV START', fold, recipe['spec']['name'], flush=True)
            pv, ps, d = train(recipe['spec'], a,y[fit],b,y[valid],c,seed=42+fold)
            np.savez_compressed(path,valid_index=valid,validation_prediction=pv,test_prediction=ps)
            write_json(path.with_suffix('.json'),d)
            del a,b,c
        with np.load(path) as z, np.load(old_dir/f'fold_{fold:02d}_xgb.npz') as bx:
            np.testing.assert_array_equal(z['valid_index'],valid)
            np.testing.assert_array_equal(bx['valid_index'],valid)
            oof[valid] = combine(old.prediction.to_numpy()[valid], z['validation_prediction'], recipe,
                                 bx['validation_prediction'])
            # Base test blend is already the mean of its ten fold predictions.
            w=recipe['weight']
            if recipe['mode']=='replace_xgb':
                prediction += w*(z['test_prediction']-bx['test_prediction'])/10
            else:
                prediction += w*z['test_prediction']/10
        coverage[valid] += 1
        scores.append(float(roc_auc_score(y[valid],oof[valid])))
        baseline.append(float(roc_auc_score(y[valid],old.prediction.iloc[valid])))
        details.append(read(path.with_suffix('.json')))
        print('CV RESULT',fold,scores[-1],'gain',scores[-1]-baseline[-1],flush=True)
        gc.collect()
    prediction += old_test.satisfaction.to_numpy()*(1 if recipe['mode']=='replace_xgb' else 1-recipe['weight'])
    assert (coverage==1).all()
    for p in [oof,prediction]:
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    old_test['satisfaction']=prediction
    old_test.to_csv(ROOT/'submission_v4.csv',index=False)
    old['prediction']=oof
    old.to_csv(ROOT/'submission_v4_oof.csv',index=False)
    result=dict(oof_auc=float(roc_auc_score(y,oof)),baseline_oof_auc=read(ROOT/'submission_v3_metrics.json')['oof_auc'],
        mean_auc=float(np.mean(scores)),std_auc=float(np.std(scores)),fold_scores=scores,
        baseline_fold_scores=baseline,improved_folds=int((np.array(scores)>baseline).sum()),
        n_splits=10,every_row_oof_once=True,directory=str(directory),details=details,
        kaggle_score=None,submitted=False)
    write_json(ROOT/'submission_v4_metrics.json',result)
    verify(directory, recipe)
    return result


def verify(directory, recipe):
    baseline=pd.read_csv(ROOT/'submission_v3_oof.csv');oof=pd.read_csv(ROOT/'submission_v4_oof.csv')
    old_test=pd.read_csv(ROOT/'submission_v3.csv');test=pd.read_csv(ROOT/'submission_v4.csv')
    template=pd.read_csv(ROOT/'sample_submission.csv');train_data=pd.read_csv(ROOT/'train.csv',usecols=['id','satisfaction'])
    actual_test=pd.read_csv(ROOT/'test.csv',usecols=['id'])
    assert test.shape==template.shape and list(test)==list(template)==['id','satisfaction']
    assert test.id.equals(template.id) and test.id.equals(actual_test.id) and test.id.is_unique
    assert oof.id.equals(train_data.id) and np.array_equal(oof.target,train_data.satisfaction.astype(int))
    assert oof.fold.equals(baseline.fold)
    pv=np.zeros(len(oof));pt=np.zeros(len(test));covered=np.zeros(len(oof),int)
    bxdir=Path(read(ROOT/'submission_v3_metrics.json')['directory'])
    for fold in range(1,11):
        with np.load(directory/f'fold_{fold:02d}.npz') as z,np.load(bxdir/f'fold_{fold:02d}_xgb.npz') as bx:
            idx=np.flatnonzero(oof.fold.to_numpy()==fold)
            np.testing.assert_array_equal(idx,z['valid_index']);np.testing.assert_array_equal(idx,bx['valid_index'])
            pv[idx]=combine(baseline.prediction.to_numpy()[idx],z['validation_prediction'],recipe,bx['validation_prediction'])
            pt+=recipe['weight']*(z['test_prediction']-(bx['test_prediction'] if recipe['mode']=='replace_xgb' else 0))/10
            covered[idx]+=1
        assert not {'id','satisfaction'}.intersection(read(directory/f'fold_{fold:02d}.json')['features'])
    pt+=old_test.satisfaction.to_numpy()*(1 if recipe['mode']=='replace_xgb' else 1-recipe['weight'])
    assert (covered==1).all()
    np.testing.assert_allclose(pv,oof.prediction,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(pt,test.satisfaction,rtol=1e-12,atol=1e-12)
    for p in [pv,pt]:assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    metrics=read(ROOT/'submission_v4_metrics.json')
    assert abs(roc_auc_score(oof.target,pv)-metrics['oof_auc'])<1e-12
    setup()
    write_json(ROOT/'v4_verification.json',dict(predictions_reconstructed=True,every_row_oof_once=True,
        inputs_and_previous_results_unchanged=True,rows=len(test),oof_auc=metrics['oof_auc'],
        submission_sha256=sha(ROOT/'submission_v4.csv')))
    print('VERIFIED',metrics['oof_auc'],flush=True)


def report(directory, recipe, confirmation=None, metrics=None):
    rows='\n'.join(f"| {a['group']} | {a['auc']:.9f} | {a['change_without_group']:+.9f} |" for a in recipe['ablations'])
    trials='\n'.join(f"| {t['spec']['name']} | {t['mode']} | {t['weight']:.1f} | {t['auc']:.9f} | {t['gain']:+.9f} |" for t in recipe['trials'])
    text=f'''# 第四轮：特征消融与模型互补验证

当前线上基线：0.96074（用户反馈，按最近交付的 v3 文件记录，未独立查询 Kaggle）。本轮保留原模型、输入及预测。

## 特征消融

完整 66 特征 XGBoost 对照 AUC：{recipe['control_auc']:.9f}；上一轮对应 AUC：{recipe['prior_xgb_auc']:.9f}。
各组使用相同参数、划分与随机种子，分别早停，因此分数反映去掉特征组后的整体训练效果。正差值表示去掉该组更好；不能把这些差值相加。

| 去掉的特征组 | 筛选 AUC | 相对完整对照变化 |
|---|---:|---:|
{rows}

## 固定候选与融合

除消融外，测试服务实际评分减辅助期望评分的 13 个残差，以及三种叶子数的 LightGBM。只使用比赛训练标签和已核验的公开原始数据辅助预测，没有导入他人预测。输入中排除 id 和 satisfaction。
LightGBM 使用完整 66 特征，增加树生长策略的多样性；并非增加新外部信息。
筛选基线是完整 v3 融合，AUC {recipe['baseline_selection_auc']:.9f}。候选和权重网格事先固定，最终只选择一个新增模型。

| 候选 | 融合方式 | 权重 | 筛选 AUC | 相对 v3 |
|---|---|---:|---:|---:|
{trials}

冻结选择：{recipe['spec']['name']}，{recipe['mode']}，权重 {recipe['weight']:.1f}，筛选提升 {recipe['selection_gain']:+.9f}。
replace_xgb 表示仅替换 v3 中权重 0.4 的新 XGBoost，保留旧融合和 RealMLP；mix 表示与完整 v3 加权平均。

## 复核与边界

筛选使用既有 fit/selection 分区。冻结之后才评估既有 development/audit 分区，固定轮数为筛选最佳轮数的 1.1 倍向上取整，不使用 audit 标签早停或调整权重。
复核集此前已被看过，本轮筛选也比较多个候选，均可能使估计偏乐观；这些结果不能当作新的独立测试分数。
预设门槛：筛选和复核相对 v3 均至少提高 0.00002，才进行完整 10 折；未过门槛就保留 v3。
'''
    if confirmation:
        text+=f"\n复核 AUC {confirmation['auc']:.9f}，v3 {confirmation['baseline_auc']:.9f}，变化 **{confirmation['gain']:+.9f}**。通过门槛：{confirmation['passed']}。\n"
    if metrics:
        text+=f"\n完整同划分 10 折 OOF：**{metrics['oof_auc']:.9f}**，v3 {metrics['baseline_oof_auc']:.9f}，变化 {metrics['oof_auc']-metrics['baseline_oof_auc']:+.9f}；{metrics['improved_folds']}/10 折提高。\n"
        text+='\n已生成并独立重建核验 submission_v4.csv 和 submission_v4_oof.csv；尚无本轮线上成绩。折模型使用外层验证集早停，本地结果仍存在选择偏差。\n'
    else:
        text+='\n尚未生成新的完整 10 折提交文件，推荐保留已验证的 submission_v3.csv。\n'
    text+=f'\n实验缓存与逐模型轮数记录：`{directory}`。复现入口：`.venv/Scripts/python.exe optimize_v4.py all`。\n'
    (ROOT/'V4_EXPERIMENT_REPORT.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['all','screen','confirmation','cv','report','verify'],default='all',nargs='?')
    args=parser.parse_args();directory=setup()
    if args.stage in ['all','screen']:
        recipe=screen(directory)
    else:
        recipe=read(directory/'recipe.json')
    conf=None;metrics=None
    report(directory,recipe)
    if args.stage in ['all','confirmation'] and recipe['selection_gain']>=.00002:
        conf=confirm(directory,recipe);report(directory,recipe,conf)
    elif (directory/'confirmation.json').exists():
        conf=read(directory/'confirmation.json')
    if args.stage in ['all','cv'] and conf and conf['passed']:
        metrics=cv(directory,recipe)
    elif (ROOT/'submission_v4_metrics.json').exists():
        m=read(ROOT/'submission_v4_metrics.json')
        if m['directory']==str(directory):metrics=m
    if args.stage=='verify':verify(directory,recipe)
    report(directory,recipe,conf,metrics)
    print('ALL COMPLETE',flush=True)


if __name__=='__main__':
    main()
