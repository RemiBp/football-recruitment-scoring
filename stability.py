"""Chronological model comparison and player-cluster stability checks."""
from __future__ import annotations
import json, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from build_dataset import NUMERIC, CATEGORICAL, FEATURES, ROOT, validate

OUT=ROOT/'reports'
SEEDS=[11,23,42,67,101]


def transform_numeric(x):
    x=np.asarray(x,dtype=float).copy()
    # Positive monetary features are highly skewed. Other columns retain units.
    for i in [NUMERIC.index('valuation_eur'),NUMERIC.index('club_mean_valuation_eur')]:
        if x.shape[1]==len(NUMERIC): x[:,i]=np.log1p(np.maximum(x[:,i],0))
    return x


def pipeline(model='Logistic',seed=42):
    numeric=Pipeline([('log',FunctionTransformer(transform_numeric)),
                      ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    pre=ColumnTransformer([('num',numeric,NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],verbose_feature_names_out=False)
    if model=='Logistic': est=LogisticRegression(C=1,max_iter=1500,random_state=seed)
    elif model=='Random Forest': est=RandomForestClassifier(n_estimators=150,min_samples_leaf=12,max_features=.8,n_jobs=2,random_state=seed)
    elif model=='XGBoost': est=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.04,subsample=.8,colsample_bytree=.8,
                                            reg_lambda=5,min_child_weight=8,n_jobs=2,random_state=seed,eval_metric='logloss')
    else: raise ValueError(model)
    return Pipeline([('pre',pre),('model',est)])


def matrix(df):
    x=df[FEATURES].copy()
    for col in CATEGORICAL: x[col]=x[col].fillna('Unknown').astype(object)
    return x


def metric(y,p):
    return {'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'n':len(y),'positive_rate':float(np.mean(y))}


def cluster_indices(groups,rng):
    unique=np.unique(groups)
    lookup={k:np.flatnonzero(groups==k) for k in unique}
    sampled=rng.choice(unique,size=len(unique),replace=True)
    return np.concatenate([lookup[k] for k in sampled])


def psi(reference,current,bins=10):
    """Training-derived bins, explicit missing bucket, add-0.5 count smoothing."""
    ref=pd.Series(reference);cur=pd.Series(current)
    if pd.api.types.is_numeric_dtype(ref):
        edges=np.unique(ref.dropna().quantile(np.linspace(0,1,bins+1)).to_numpy())
        edges=np.r_[-np.inf,edges[1:-1],np.inf]
        if len(edges)<2: edges=np.array([-np.inf,np.inf])
        def counts(s):
            return np.r_[np.histogram(s.dropna(),bins=edges)[0],s.isna().sum()]
        r,c=counts(ref),counts(cur)
    else:
        levels=sorted(ref.dropna().astype(str).unique().tolist())
        def counts(s):
            ss=s.astype('string');return np.array([(ss==v).sum() for v in levels]+[(ss.notna() & ~ss.isin(levels)).sum(),ss.isna().sum()])
        r,c=counts(ref),counts(cur)
    r=(r+.5)/(sum(r)+.5*len(r));c=(c+.5)/(sum(c)+.5*len(c))
    return float(np.sum((c-r)*np.log(c/r)))


def feature_names(fit):
    cats=fit.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL).tolist()
    return NUMERIC+cats


def wilson(k,n):
    if not n:return (None,None)
    z=1.96;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h


def group_audit(test, predictions):
    rows=[]
    for model,p in predictions.items():
        temp=test[['citizenship_audit','position','target_10']].copy();temp['prediction']=p>=.5
        for controlled in [False,True]:
            keys=['citizenship_audit','position'] if controlled else ['citizenship_audit']
            for key,s in temp.groupby(keys,dropna=False):
                if len(s)<(25 if controlled else 50):continue
                country=key[0] if isinstance(key,tuple) else key
                y=s.target_10.astype(int).to_numpy();pr=s.prediction.to_numpy()
                neg=int((y==0).sum());pos=int((y==1).sum());fp=int(((y==0)&pr).sum());fn=int(((y==1)&~pr).sum())
                if min(neg,pos)<5:continue
                fl,fu=wilson(fp,neg);nl,nu=wilson(fn,pos)
                rows.append(dict(model=model,citizenship=str(country),position=key[1] if controlled else 'All',
                                 n=len(s),positives=pos,negatives=neg,fpr=fp/neg,fnr=fn/pos,
                                 fpr_low=fl,fpr_high=fu,fnr_low=nl,fnr_high=nu))
    return pd.DataFrame(rows)


def run(bootstraps=200):
    df=pd.read_parquet(ROOT/'dataset_final.parquet');validate(df)
    labeled=df[df.outcome_observed].copy()
    train=labeled[labeled.split.eq('train')];val=labeled[labeled.split.eq('validation')];test=labeled[labeled.split.eq('test')]
    xtr,xv,xt=map(matrix,[train,val,test]);ytr=train.target_10.astype(int).to_numpy();yt=test.target_10.astype(int).to_numpy()
    models={};preds={};rows=[];ps=[]
    for name in ['Logistic','Random Forest','XGBoost']:
        print('Fit',name,flush=True);fit=pipeline(name).fit(xtr,ytr);models[name]=fit;preds[name]=fit.predict_proba(xt)[:,1]
        pv=fit.predict_proba(xv)[:,1];pt=fit.predict_proba(xtr)[:,1]
        for label,part,p in [('validation',val,pv),('test',test,preds[name])]:
            rows.append({'model':name,'scope':label,**metric(part.target_10.astype(int),p)})
        for year in sorted(test.target_season.unique()):
            mask=test.target_season.eq(year).to_numpy();rows.append({'model':name,'scope':str(year),**metric(yt[mask],preds[name][mask])})
        mask=~test.player_id.isin(train.player_id).to_numpy()
        rows.append({'model':name,'scope':'unseen_players',**metric(yt[mask],preds[name][mask])})
        for year in ['test']+sorted(test.target_season.unique().tolist()):
            mask=np.ones(len(test),dtype=bool) if year=='test' else test.target_season.eq(year).to_numpy()
            ps.append({'variable':'score','model':name,'scope':str(year),'reference':'train (in-sample)','psi':psi(pt,preds[name][mask])})
            ps.append({'variable':'score','model':name,'scope':str(year),'reference':'validation (out-of-time)','psi':psi(pv,preds[name][mask])})
    for col in FEATURES:
        for year in ['test']+sorted(test.target_season.unique().tolist()):
            m=np.ones(len(test),dtype=bool) if year=='test' else test.target_season.eq(year).to_numpy()
            ps.append({'variable':col,'model':'input','scope':str(year),'reference':'train','psi':psi(train[col],test.loc[m,col])})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(ps).to_csv(OUT/'psi.csv',index=False)
    # Conditional uncertainty of fixed models on a player-clustered test sample.
    rng=np.random.default_rng(20260922);aucs=[];coefs=[]
    reference_scale=models['Logistic'].named_steps['pre'].named_transformers_['num'].named_steps['scale'].scale_
    for b in range(bootstraps):
        if b%25==0: print('Bootstrap',b,'/',bootstraps,flush=True)
        it=cluster_indices(test.player_id.to_numpy(),rng)
        if len(np.unique(yt[it]))==2:
            for name,p in preds.items():aucs.append({'replicate':b,'model':name,'auc':roc_auc_score(yt[it],p[it])})
        ir=cluster_indices(train.player_id.to_numpy(),rng)
        fit=pipeline('Logistic',seed=b).fit(xtr.iloc[ir],ytr[ir]);coef=fit.named_steps['model'].coef_[0].copy()
        scale=fit.named_steps['pre'].named_transformers_['num'].named_steps['scale'].scale_
        coef[:len(NUMERIC)] *= reference_scale/scale  # common original-training SD units
        coefs.extend({'replicate':b,'feature':f,'coefficient':float(c)} for f,c in zip(feature_names(fit),coef))
    pd.DataFrame(aucs).to_csv(OUT/'bootstrap_auc.csv',index=False)
    cf=pd.DataFrame(coefs);cf.to_csv(OUT/'bootstrap_coefficients.csv',index=False)
    cf.groupby('feature').coefficient.agg(['count','mean','std',lambda s:s.quantile(.025),lambda s:s.quantile(.975),lambda s:(s>0).mean()]).set_axis(['replicates','mean','std','lower','upper','positive_fraction'],axis=1).to_csv(OUT/'coefficient_intervals.csv')
    seed_rows=[];seed_variation=[]
    for name in ['Random Forest','XGBoost']:
        pp=[]
        for seed in SEEDS:
            print('Seed',name,seed,flush=True);f=pipeline(name,seed).fit(xtr,ytr);p=f.predict_proba(xt)[:,1];pp.append(p)
            seed_rows.append({'model':name,'seed':seed,**metric(yt,p)})
        pp=np.array(pp);seed_variation.append({'model':name,'mean_score_sd':float(pp.std(axis=0).mean()),
                                             'p95_score_range':float(np.quantile(np.ptp(pp,axis=0),.95)),
                                             'auc_range':float(np.ptp([roc_auc_score(yt,p) for p in pp]))})
    pd.DataFrame(seed_rows).to_csv(OUT/'seeds.csv',index=False);pd.DataFrame(seed_variation).to_csv(OUT/'seed_variation.csv',index=False)
    # Common 1% SD noise on continuous input measurements. Counts and categories
    # are unchanged; valuation growth is recomputed consistently with prior price.
    noisy=xt.copy();rng=np.random.default_rng(17)
    for col in ['age','height_cm','valuation_eur','club_mean_valuation_eur']:
        original=noisy[col].copy();noisy[col]=(original+rng.normal(0,.01*xtr[col].std(),len(xt))).clip(lower=0)
        if col=='valuation_eur':
            ratio=noisy[col]/original.replace(0,np.nan)
            noisy['valuation_growth_12m']=(1+noisy.valuation_growth_12m)*ratio-1
    perturb=[]
    for name,f in models.items():
        p=f.predict_proba(noisy)[:,1];base=preds[name]
        perturb.append({'model':name,'auc_before':roc_auc_score(yt,base),'auc_after':roc_auc_score(yt,p),
                        'mean_absolute_score_change':float(np.abs(p-base).mean()),
                        'p95_absolute_score_change':float(np.quantile(np.abs(p-base),.95)),
                        'decision_flip_rate_at_0_5':float(((p>=.5)!=(base>=.5)).mean())})
    pd.DataFrame(perturb).to_csv(OUT/'perturbations.csv',index=False)
    sensitivity=[]
    for threshold in [8,10,15]:
        y=train[f'target_{threshold}'].astype(int).to_numpy();yy=test[f'target_{threshold}'].astype(int).to_numpy()
        for name in models:
            f=models[name] if threshold==10 else pipeline(name).fit(xtr,y)
            sensitivity.append({'threshold':threshold,'model':name,'train_positive_rate':float(y.mean()),**metric(yy,f.predict_proba(xt)[:,1])})
    pd.DataFrame(sensitivity).to_csv(OUT/'target_sensitivity.csv',index=False)
    # A valuation ablation is associational. Removing a feature is not a causal
    # identification strategy for nationality discrimination.
    abtrain=xtr.copy();abtest=xt.copy()
    for c in ['valuation_eur','valuation_growth_12m','club_mean_valuation_eur']: abtrain[c]=0.;abtest[c]=0.
    ab=pipeline('Logistic').fit(abtrain,ytr);preds['Logistic without valuation']=ab.predict_proba(abtest)[:,1]
    group_audit(test,preds).to_csv(OUT/'fairness_descriptive.csv',index=False)
    pd.DataFrame([{'model':'Logistic without valuation','scope':'test',**metric(yt,preds['Logistic without valuation'])}]).to_csv(OUT/'valuation_ablation.csv',index=False)
    pred_frame=test[['player_id','target_season','target_10']].copy()
    for name,p in preds.items():pred_frame[name]=p
    # Kept locally only. The public report is aggregate, not a shortlist of people.
    (ROOT/'models').mkdir(exist_ok=True);pred_frame.to_parquet(ROOT/'models/test_predictions.parquet',index=False)
    summary={'bootstraps':bootstraps,'cluster_unit':'player_id','seed_repetitions':SEEDS,
             'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
             'train_positive_rate':float(ytr.mean()),'test_positive_rate':float(yt.mean()),
             'missing_outcome_fraction':float((~df.outcome_observed).mean()),
             'train_goals_quantiles':train.goals_next.quantile([.25,.5,.75,.9]).to_dict(),
             'threshold_policy':'10 primary, 8 and 15 sensitivity; no data-driven threshold selection',
             'parameters':{n:models[n].named_steps['model'].get_params() for n in models}}
    (OUT/'run.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bootstraps',type=int,default=200);args=p.parse_args();run(args.bootstraps)
