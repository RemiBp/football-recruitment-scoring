"""Write and execute a small review notebook without rerunning model training."""
import contextlib,io,json
import nbformat as nbf
from pathlib import Path
root=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nb.cells=[nbf.v4.new_markdown_cell('# Football scoring: dataset audit\n\nRun from the repository root. Results below come from the released table and cloud outputs, not illustrative data.'),
nbf.v4.new_code_cell("import pandas as pd\nfrom build_dataset import ROOT, validate, FEATURES\ndf = pd.read_parquet(ROOT / 'dataset_final.parquet')\nvalidate(df)\nprint(f'{len(df):,} rows; {df.player_id.nunique():,} players; {df.outcome_observed.sum():,} observed outcomes')\nprint(df.groupby('split').agg(rows=('player_id','size'), observed=('outcome_observed','sum')).to_string())"),
nbf.v4.new_markdown_cell('## Strict valuation cutoff\nEvery observed valuation date must precede July 1. Missing valuations are counted separately. This check does not certify the historical validity of profile metadata.'),
nbf.v4.new_code_cell("observed = df.valuation_date.notna()\nassert (df.loc[observed, 'valuation_date'] < df.loc[observed, 'cutoff_date']).all()\nprint(f'{observed.sum():,}/{observed.sum():,} dated valuations are strictly before cutoff')\nprint(f'Missing valuations: {(~observed).sum()}')\nprint('Missing outcomes:', df.target_10.isna().sum())\nassert df.loc[~df.outcome_observed, 'target_10'].isna().all()\nprint('Nationality excluded from model:', 'citizenship_audit' not in FEATURES)"),
nbf.v4.new_markdown_cell('## Future-data poisoning test\nAdding a valuation on or after the cutoff must not alter the selected historical valuation.'),
nbf.v4.new_code_cell("from tests.test_integrity import test_cutoff_and_future_poisoning\ntest_cutoff_and_future_poisoning()\nprint('Passed: exact-cutoff and future valuations are excluded')"),
nbf.v4.new_markdown_cell('## Out-of-time model performance\nRead the complete CSVs for validation, test seasons and unseen-player sensitivity. TabICL uses a smaller training context; matched baselines are provided separately.'),
nbf.v4.new_code_cell("m = pd.read_csv(ROOT / 'reports/metrics.csv')\nf = ROOT / 'reports/foundation_metrics.csv'\nif f.exists(): m = pd.concat([m, pd.read_csv(f)], ignore_index=True)\nprint(m[m.scope.eq('test')].to_string(index=False))\nb = pd.read_csv(ROOT / 'reports/bootstrap_auc.csv')\nprint(b.groupby('model').auc.agg(['count', 'mean', lambda x: x.quantile(.025), lambda x: x.quantile(.975)]).to_string())"),
nbf.v4.new_markdown_cell('## Target sensitivity\nThe positive rate is measured; it is not assumed to be 20–25%. Quartiles below use training outcomes only. The main threshold remains the preregistered 10 goals.'),
nbf.v4.new_code_cell("print(df.loc[df.split.eq('train'), 'goals_next'].quantile([.25,.5,.75,.9]).to_string())\nprint(pd.read_csv(ROOT / 'reports/target_sensitivity.csv').to_string(index=False))"),
nbf.v4.new_markdown_cell('## Limits\nMissing next-season coverage is not a negative label. National-team totals are unavailable. Current profile metadata, movement outside the observed leagues, stale valuations and repeated players limit generalisation. Bootstrap intervals use player clusters. Nationality comparisons are descriptive, not a causal discrimination finding.')]
scope={}
for i,c in enumerate([c for c in nb.cells if c.cell_type=='code'],start=1):
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):exec(c.source,scope)
    c.execution_count=i;c.outputs=[nbf.v4.new_output('stream',name='stdout',text=stream.getvalue())]
nbf.validate(nb);nbf.write(nb,root/'audit.ipynb')
print('Executed audit.ipynb')
