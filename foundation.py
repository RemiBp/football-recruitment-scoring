"""TabICL comparator with an explicit bounded context and matched-data baselines."""
import json, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier
from stability import matrix, pipeline, metric, cluster_indices, psi
from build_dataset import ROOT


def run():
    torch.set_num_threads(2)
    df=pd.read_parquet(ROOT/'dataset_final.parquet');df=df[df.outcome_observed]
    train=df[df.split.eq('train')];test=df[df.split.eq('test')];val=df[df.split.eq('validation')]
    # Fixed, train-only stratified context sample; the whole chronological test is
    # evaluated. Classical comparators on this same subset make the cap visible.
    if len(train)>1500:
        ids,_=train_test_split(np.arange(len(train)),train_size=1500,random_state=42,stratify=train.target_10)
        train=train.iloc[ids].sort_values(['target_season','player_id'])
    xtr=matrix(train);xt=matrix(test);xv=matrix(val);y=train.target_10.astype(int).to_numpy()
    pre=pipeline().named_steps['pre'];xx=pre.fit_transform(xtr);tt=pre.transform(xt);vv=pre.transform(xv)
    model=TabICLClassifier(n_estimators=1,device='cpu',n_jobs=2,random_state=42,
                          checkpoint_version='tabicl-classifier-v2-20260212.ckpt',verbose=True)
    start=time.monotonic();print('TabICL fit',xx.shape,flush=True);model.fit(xx,y)
    def predict(x):
        pieces=[]
        for i in range(0,len(x),256):
            print('TabICL predict',i,'/',len(x),flush=True)
            pieces.append(model.predict_proba(x[i:i+256])[:,1])
        return np.concatenate(pieces)
    pt=predict(tt);pv=predict(vv)
    rows=[{'model':'TabICL','scope':'test',**metric(test.target_10.astype(int),pt)},
          {'model':'TabICL','scope':'validation',**metric(val.target_10.astype(int),pv)}]
    for year in sorted(test.target_season.unique()):
        m=test.target_season.eq(year).to_numpy();rows.append({'model':'TabICL','scope':str(year),**metric(test.target_10.to_numpy()[m].astype(int),pt[m])})
    unseen=~test.player_id.isin(df[df.split.eq('train')].player_id).to_numpy()
    rows.append({'model':'TabICL','scope':'unseen_players',**metric(test.target_10.to_numpy()[unseen].astype(int),pt[unseen])})
    pd.DataFrame(rows).to_csv(ROOT/'reports/foundation_metrics.csv',index=False)
    matched=[]
    for name in ['Logistic','Random Forest','XGBoost']:
        f=pipeline(name).fit(xtr,y)
        matched.append({'model':name,'scope':'test_matched_1500_context',**metric(test.target_10.astype(int),f.predict_proba(xt)[:,1])})
    pd.DataFrame(matched).to_csv(ROOT/'reports/foundation_matched_baselines.csv',index=False)
    rng=np.random.default_rng(20260922);boots=[];yt=test.target_10.astype(int).to_numpy()
    for b in range(200):
        idx=cluster_indices(test.player_id.to_numpy(),rng)
        boots.append({'replicate':b,'model':'TabICL','auc':metric(yt[idx],pt[idx])['auc']})
    pd.DataFrame(boots).to_csv(ROOT/'reports/foundation_bootstrap_auc.csv',index=False)
    ps=[]
    for scope in ['test']+sorted(test.target_season.unique().tolist()):
        m=np.ones(len(test),dtype=bool) if scope=='test' else test.target_season.eq(scope).to_numpy()
        ps.append({'variable':'score','model':'TabICL','scope':str(scope),'reference':'validation (out-of-time)','psi':psi(pv,pt[m])})
    pd.DataFrame(ps).to_csv(ROOT/'reports/foundation_psi.csv',index=False)
    (ROOT/'reports/foundation_run.json').write_text(json.dumps({'model':'TabICL','checkpoint':'tabicl-classifier-v2-20260212.ckpt',
        'context_rows':len(train),'context_sampling':'training-only, stratified on target, seed42',
        'ensemble_size':1,'inference_batch_size':256,'device':'cpu','seconds':time.monotonic()-start,
        'comparison_limit':'Classical primary models use all training rows; matched 1500-row baselines are provided separately.'},indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':run()
