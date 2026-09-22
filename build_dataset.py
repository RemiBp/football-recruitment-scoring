"""Build the frozen player-season table from the public Transfermarkt snapshot."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
URL = 'https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores'
LEAGUES = ['ES1', 'FR1', 'GB1', 'IT1', 'L1', 'NL1', 'PO1', 'RU1', 'TR1']
OFFENSIVE = ['Centre-Forward', 'Left Winger', 'Right Winger', 'Second Striker', 'Attacking Midfield']
NUMERIC = ['goals_prev', 'assists_prev', 'minutes_prev', 'goals_per90_prev', 'appearances_prev',
           'age', 'valuation_eur', 'valuation_growth_12m', 'club_mean_valuation_eur',
           'height_cm', 'club_change_known_at_cutoff']
CATEGORICAL = ['position', 'league', 'foot']
FEATURES = NUMERIC + CATEGORICAL


def strict_asof(left, right, left_on, right_on, by='player_id'):
    """Backward-only join; values dated exactly at the information cutoff are excluded."""
    return pd.merge_asof(left.sort_values(left_on), right.sort_values(right_on),
                         left_on=left_on, right_on=right_on, by=by,
                         direction='backward', allow_exact_matches=False)


def make_labels(goals):
    return pd.DataFrame({f'target_{t}': (goals.ge(t).astype('Int8').mask(goals.isna()))
                         for t in [8, 10, 15]})


def load(raw, name, cols):
    print('Read', name, flush=True)
    return pd.read_csv(raw / f'{name}.csv', usecols=cols)


def build(raw: Path, output: Path):
    g = load(raw, 'games', ['game_id','season','date','competition_id','competition_type',
                            'home_club_goals','away_club_goals'])
    g['date'] = pd.to_datetime(g['date'])
    a = load(raw, 'appearances', ['game_id','player_id','player_club_id','goals','assists',
                                'minutes_played','yellow_cards','red_cards'])
    assert not a.duplicated(['game_id','player_id']).any(), 'duplicate appearances'
    a = a.merge(g, on='game_id', validate='many_to_one')
    # Single-phase July-June leagues with long historical coverage. Postponed COVID
    # seasons cannot meet a July 1 information boundary and are excluded explicitly.
    valid_g = g[g.competition_id.isin(LEAGUES) & g.season.between(2013,2025)].copy()
    coverage = valid_g.groupby(['competition_id','season']).agg(
        games=('game_id','size'), first=('date','min'), last=('date','max'))
    coverage['valid'] = ((coverage['first'] >= pd.to_datetime(coverage.index.get_level_values('season').astype(str)+'-07-01')) &
                         (coverage['last'] < pd.to_datetime((coverage.index.get_level_values('season')+1).astype(str)+'-07-01')) &
                         (coverage.games >= 200))
    coverage.loc[coverage.index.get_level_values('season') == 2019, 'valid'] = False
    coverage.reset_index().to_csv(ROOT/'reports/league_coverage.csv',index=False)
    good_keys = coverage[coverage.valid].reset_index()[['competition_id','season']]
    d = a.merge(good_keys, on=['competition_id','season'], how='inner')
    assert (d.minutes_played >= 0).all()
    # Match positions, not today's profile position. Minute-weighted modal position.
    lineup = load(raw,'game_lineups',['game_id','player_id','position'])
    lineup = lineup.drop_duplicates(['game_id','player_id'])
    pos = d[['game_id','player_id','season','minutes_played']].merge(lineup,on=['game_id','player_id'],how='inner')
    pos = pos.groupby(['player_id','season','position'],as_index=False).minutes_played.sum()
    pos = pos.sort_values(['minutes_played','position']).drop_duplicates(['player_id','season'],keep='last')
    agg = d.groupby(['player_id','season'],as_index=False).agg(
        goals_prev=('goals','sum'), assists_prev=('assists','sum'), minutes_prev=('minutes_played','sum'),
        appearances_prev=('game_id','nunique'), yellow_cards_prev=('yellow_cards','sum'),
        red_cards_prev=('red_cards','sum'), performance_max_date=('date','max'))
    club = d.groupby(['player_id','season','player_club_id','competition_id'],as_index=False).minutes_played.sum()
    club = club.sort_values(['minutes_played','player_club_id']).drop_duplicates(['player_id','season'],keep='last')
    agg = agg.merge(club[['player_id','season','player_club_id','competition_id']],on=['player_id','season'],validate='one_to_one')
    agg = agg.merge(pos[['player_id','season','position']],on=['player_id','season'],how='left',validate='one_to_one')
    squad = agg.copy()
    df = agg[(agg.minutes_prev >= 900) & agg.position.isin(OFFENSIVE)].copy()
    df['target_season'] = df.season + 1
    df = df[df.target_season.between(2014,2025) & ~df.target_season.isin([2019,2020])].copy()
    df = df.rename(columns={'season':'prior_season','competition_id':'league','player_club_id':'prior_club_id'})
    df['cutoff_date'] = pd.to_datetime(df.target_season.astype(str)+'-07-01')
    df['cutoff_minus_year'] = pd.to_datetime((df.target_season-1).astype(str)+'-07-01')
    p = load(raw,'players',['player_id','country_of_citizenship','date_of_birth','height_in_cm','foot'])
    p['date_of_birth'] = pd.to_datetime(p.date_of_birth)
    p = p.rename(columns={'country_of_citizenship':'citizenship_audit','height_in_cm':'height_cm'})
    df = df.merge(p,on='player_id',how='left',validate='many_to_one')
    df['age'] = (df.cutoff_date-df.date_of_birth).dt.days / 365.25
    df = df[df.age.ge(18)].copy()
    df['goals_per90_prev'] = 90 * df.goals_prev / df.minutes_prev
    v = load(raw,'player_valuations',['player_id','date','market_value_in_eur'])
    v['date'] = pd.to_datetime(v.date)
    assert not v.duplicated(['player_id','date']).any(), 'ambiguous valuation date'
    v = v.rename(columns={'date':'valuation_date','market_value_in_eur':'valuation_eur'})
    df = strict_asof(df,v,'cutoff_date','valuation_date')
    earlier = v.rename(columns={'valuation_date':'valuation_prior_date','valuation_eur':'valuation_prior_eur'})
    df = strict_asof(df,earlier,'cutoff_minus_year','valuation_prior_date')
    df['valuation_growth_12m'] = df.valuation_eur / df.valuation_prior_eur.replace(0,np.nan) - 1
    # Reconstruct prior-season squad prices at each cutoff. No current club totals.
    squad['cutoff_date'] = pd.to_datetime((squad.season+1).astype(str)+'-07-01')
    squad = strict_asof(squad,v,'cutoff_date','valuation_date')
    cm = squad.groupby(['player_club_id','season'],as_index=False).agg(
        club_mean_valuation_eur=('valuation_eur','mean'), club_valued_players=('valuation_eur','count'),
        club_squad_players=('player_id','size'), club_valuation_max_date=('valuation_date','max'))
    cm = cm.rename(columns={'player_club_id':'prior_club_id','season':'prior_season'})
    df = df.merge(cm,on=['prior_club_id','prior_season'],how='left',validate='many_to_one')
    # Observed tournament matches only, not full career caps. Missing coverage is
    # explicitly documented; zero means no observed caps in this source.
    intl = a[a.competition_type.eq('national_team_competition')].copy()
    intl = intl.groupby(['player_id','date'],as_index=False).agg(caps=('game_id','nunique'),intl_goals=('goals','sum'))
    intl = intl.sort_values(['player_id','date'])
    intl['observed_international_appearances'] = intl.groupby('player_id').caps.cumsum()
    intl['observed_international_goals'] = intl.groupby('player_id').intl_goals.cumsum()
    intl = intl.rename(columns={'date':'international_max_date'})
    df = strict_asof(df,intl[['player_id','international_max_date','observed_international_appearances','observed_international_goals']],
                     'cutoff_date','international_max_date')
    for col in ['observed_international_appearances','observed_international_goals']:
        # This snapshot has no matched national-team appearances. Missing must
        # remain unavailable, not an invented career-cap total.
        df[col] = df[col].astype(float)
    t = load(raw,'transfers',['player_id','transfer_date','from_club_id','to_club_id'])
    t['transfer_date'] = pd.to_datetime(t.transfer_date)
    # Multiple transfers on one date can be an administrative loan return/reloan.
    # Do not arbitrarily pick one destination in such cases.
    t['transfer_ambiguous'] = t.duplicated(['player_id','transfer_date'],keep=False)
    t = t.sort_values(['player_id','transfer_date','to_club_id']).drop_duplicates(['player_id','transfer_date'],keep='last')
    df = strict_asof(df,t,'cutoff_date','transfer_date')
    df['club_change_known_at_cutoff'] = ((df.transfer_date > df.performance_max_date) &
                                        (df.to_club_id != df.prior_club_id)).astype(float)
    df.loc[df.transfer_ambiguous.fillna(False) & (df.transfer_date > df.performance_max_date),
           'club_change_known_at_cutoff'] = np.nan
    # An observed domestic season is needed for a label. Absence never becomes 0.
    target = d.groupby(['player_id','season'],as_index=False).agg(
        goals_next=('goals','sum'), target_minutes=('minutes_played','sum'),
        target_first_date=('date','min'), target_last_date=('date','max'))
    target = target.rename(columns={'season':'target_season'})
    df = df.merge(target,on=['player_id','target_season'],how='left',validate='one_to_one')
    df['outcome_observed'] = df.goals_next.notna()
    df = pd.concat([df,make_labels(df.goals_next)],axis=1)
    df['split'] = np.select([df.target_season.le(2021),df.target_season.eq(2022)],['train','validation'],default='test')
    df['top_five_league'] = df.league.isin(['ES1','FR1','GB1','IT1','L1'])
    # Freeze ordered schema and explicit types. Names, URLs and images are omitted.
    columns = ['player_id','prior_season','target_season','split','cutoff_date'] + FEATURES + [
        'citizenship_audit','prior_club_id','top_five_league','yellow_cards_prev','red_cards_prev',
        'performance_max_date','valuation_date','valuation_prior_date','club_valuation_max_date',
        'club_valued_players','club_squad_players','observed_international_appearances',
        'observed_international_goals','international_max_date','transfer_date',
        'outcome_observed','goals_next','target_minutes','target_first_date','target_last_date',
        'target_8','target_10','target_15']
    df = df[columns].sort_values(['target_season','player_id']).reset_index(drop=True)
    for col in NUMERIC + ['goals_next','target_minutes']: df[col] = df[col].astype('float64')
    for col in CATEGORICAL + ['citizenship_audit','split']: df[col] = df[col].astype('string')
    for col in ['player_id','prior_season','target_season','prior_club_id','yellow_cards_prev','red_cards_prev','club_valued_players','club_squad_players']: df[col] = df[col].astype('int64')
    validate(df)
    df.to_parquet(output,index=False)
    schema = {'version':'1.0.0','columns':{c:str(df[c].dtype) for c in df},'features':FEATURES,
              'audit_only':['citizenship_audit'],'target':'target_10'}
    (ROOT/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    manifest = {'source':URL,'publisher_snapshot':'2026-07-06','schema_version':'1.0.0',
                'leagues':LEAGUES,'rows':len(df),'players':int(df.player_id.nunique()),
                'observed_labels':int(df.outcome_observed.sum()),
                'dataset_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                'raw_sha256':{f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in sorted(raw.glob('*.csv'))}}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    df.groupby(['split','target_season']).agg(rows=('player_id','size'),observed=('outcome_observed','sum'),
                                             positive_rate=('target_10','mean')).to_csv(ROOT/'reports/cohort.csv')
    print(json.dumps({k:v for k,v in manifest.items() if k!='raw_sha256'},indent=2),flush=True)


def validate(df):
    assert not df.duplicated(['player_id','target_season']).any()
    assert (df.minutes_prev >= 900).all()
    assert df.position.isin(OFFENSIVE).all()
    assert df.age.ge(18).all()
    for col in ['performance_max_date','valuation_date','club_valuation_max_date','international_max_date','transfer_date']:
        assert (df[col].isna() | (df[col] < df.cutoff_date)).all(), col
    earlier = pd.to_datetime((df.target_season-1).astype(str)+'-07-01')
    assert (df.valuation_prior_date.isna() | (df.valuation_prior_date < earlier)).all()
    obs = df.outcome_observed
    assert (df.loc[obs,'target_first_date'] >= df.loc[obs,'cutoff_date']).all()
    assert (df.loc[obs,'target_last_date'] < pd.to_datetime((df.loc[obs,'target_season']+1).astype(str)+'-07-01')).all()
    for t in [8,10,15]:
        assert df.loc[~obs,f'target_{t}'].isna().all()
        assert (df.loc[obs,f'target_{t}'].astype(int) == df.loc[obs,'goals_next'].ge(t).astype(int)).all()
    assert df[df.split.eq('train')].target_season.max() < df[df.split.eq('validation')].target_season.min()
    assert df[df.split.eq('validation')].target_season.max() < df[df.split.eq('test')].target_season.min()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=ROOT/'dataset_final.parquet');parser.add_argument('--download',action='store_true')
    args=parser.parse_args();(ROOT/'reports').mkdir(exist_ok=True);args.raw.mkdir(parents=True,exist_ok=True)
    if args.download:
        archive=ROOT/'data/source.zip';urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if Path(member.filename).name != member.filename: raise ValueError('Unexpected archive path')
            z.extractall(args.raw)
    build(args.raw,args.output)
