import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from build_dataset import strict_asof, make_labels, validate, ROOT, FEATURES
from stability import psi, cluster_indices


def test_cutoff_and_future_poisoning():
    left=pd.DataFrame({'player_id':[1,1],'cutoff':pd.to_datetime(['2023-07-01','2024-07-01'])})
    past=pd.DataFrame({'player_id':[1,1],'date':pd.to_datetime(['2023-06-30','2024-06-30']),'value':[10,20]})
    poisoned=pd.concat([past,pd.DataFrame({'player_id':[1,1],'date':pd.to_datetime(['2023-07-01','2025-01-01']),'value':[999999,999999]})])
    assert strict_asof(left,poisoned,'cutoff','date').value.tolist()==[10,20]
    pd.testing.assert_frame_equal(strict_asof(left,past,'cutoff','date'),strict_asof(left,poisoned,'cutoff','date'))


def test_missing_outcome_is_not_zero():
    labels=make_labels(pd.Series([0.,9.,10.,np.nan]))
    assert labels.target_10.iloc[:3].tolist()==[0,0,1]
    assert labels.iloc[3].isna().all()


def test_psi_identical_and_shift():
    x=np.arange(1000,dtype=float)
    assert psi(x,x)==0
    assert psi(x,x+1000)>.5
    assert psi(pd.Series(['a','b',None]),pd.Series(['a','b',None]))==0
    assert psi(x,np.full(20,np.nan))>0


def test_cluster_bootstrap_keeps_all_player_seasons():
    groups=np.array([1,1,1,2,2,3]);idx=cluster_indices(groups,np.random.default_rng(42))
    counts=np.bincount(idx,minlength=len(groups))
    assert counts[0]==counts[1]==counts[2]
    assert counts[3]==counts[4]


def test_released_table_and_schema():
    import json
    data=pd.read_parquet(ROOT/'dataset_final.parquet');validate(data)
    schema=json.loads((ROOT/'schema.json').read_text())
    assert {c:str(data[c].dtype) for c in data}==schema['columns']
    assert not {'player_id','citizenship_audit','goals_next','target_10','target_season'}.intersection(FEATURES)
    assert data.valuation_date.notna().any()
    assert data.loc[data.valuation_date.notna(),'valuation_date'].lt(data.loc[data.valuation_date.notna(),'cutoff_date']).all()
