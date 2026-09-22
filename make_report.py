"""Generate a standalone HTML handoff from verified CSV outputs."""
import io,json,html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_dataset import ROOT,FEATURES

R=ROOT/'reports';out=ROOT/'docs';out.mkdir(exist_ok=True)
m=pd.read_csv(R/'metrics.csv');b=pd.read_csv(R/'bootstrap_auc.csv');run=json.loads((R/'run.json').read_text())
assert run['bootstraps']==200,'Do not publish smoke-test confidence intervals'
foundation=(R/'foundation_metrics.csv').exists()
if foundation:
 m=pd.concat([m,pd.read_csv(R/'foundation_metrics.csv')],ignore_index=True)
 b=pd.concat([b,pd.read_csv(R/'foundation_bootstrap_auc.csv')],ignore_index=True)
colors={'Logistic':'#184e38','Random Forest':'#a98039','XGBoost':'#627892','TabICL':'#905763'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.spines.left':False,'axes.spines.bottom':False,'axes.facecolor':'#f7f6f0',
                     'figure.facecolor':'#f7f6f0','axes.labelcolor':'#273d31','text.color':'#273d31'})
fig,ax=plt.subplots(figsize=(9,3.6))
for name in m.model.unique():
 s=m[(m.model==name)&m.scope.isin(['2023','2024','2025'])].sort_values('scope')
 ax.plot(s.scope,s.auc,label=name,color=colors[name],marker='o',linewidth=2)
ax.set_ylim(.65,.9);ax.set_ylabel('ROC AUC');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,ncol=2,loc='lower left');fig.tight_layout()
buf=io.StringIO();fig.savefig(buf,format='svg',bbox_inches='tight');svg=buf.getvalue().split('<svg',1)[1];svg='<svg'+svg;svg='\n'.join(line.rstrip() for line in svg.splitlines())+'\n';plt.close(fig)
(R/'temporal_auc.svg').write_text(svg)
rows=[]
for _,s in m[m.scope.eq('test')].iterrows():
 boot=b[b.model.eq(s.model)].auc;lo,hi=boot.quantile([.025,.975])
 label=s.model+(' *' if s.model=='TabICL' else '')
 rows.append(f'<tr><td>{label}</td><td>{s.auc:.3f}</td><td>{lo:.3f} to {hi:.3f}</td><td>{s.average_precision:.3f}</td><td>{s.brier:.3f}</td></tr>')
pdeltas=b[b.model.isin(['Logistic','Random Forest','XGBoost'])].pivot(index='replicate',columns='model',values='auc')
diffs=[]
for name in ['Random Forest','XGBoost']:
 d=pdeltas['Logistic']-pdeltas[name];lo,hi=d.quantile([.025,.975]);diffs.append(f'Logistic vs {name}: paired AUC difference interval {lo:+.3f} to {hi:+.3f}.')
(R/'paired_auc_comparisons.txt').write_text('\n'.join(diffs)+'\n')
pert=pd.read_csv(R/'perturbations.csv');seeds=pd.read_csv(R/'seed_variation.csv');ps=pd.read_csv(R/'psi.csv')
pscores=ps[(ps.variable=='score')&(ps.scope=='test')&(ps.reference=='validation (out-of-time)')]
stab=[]
for _,s in pert.iterrows():
 sd=seeds[seeds.model.eq(s.model)];seed='Deterministic fit' if sd.empty else f'{sd.auc_range.iloc[0]:.4f}'
 pi=pscores[pscores.model.eq(s.model)].psi.iloc[0]
 stab.append(f'<tr><td>{s.model}</td><td>{seed}</td><td>{s.mean_absolute_score_change*100:.2f} pp</td><td>{pi:.3f}</td></tr>')
cohort=pd.read_csv(R/'cohort.csv');df=pd.read_parquet(ROOT/'dataset_final.parquet')
miss=1-df.outcome_observed.mean();val=df.valuation_date.notna().sum()
featureps=ps[(ps.model=='input')&(ps.scope=='test')].sort_values('psi',ascending=False).head(3)
psitext=', '.join(f'{s.variable.replace("_"," ")} ({s.psi:.2f})' for _,s in featureps.iterrows())
foundationnote='* TabICL uses 1,500 training rows and one ensemble member. Matched classical baselines are in the repository.' if foundation else 'The TabICL comparator is still running; it is not included in these results.'
matched=''
if foundation:
 mm=pd.read_csv(R/'foundation_matched_baselines.csv');matched='<p class="small">Same 1,500-row training sample: '+', '.join(f'{s.model} AUC {s.auc:.3f}' for _,s in mm.iterrows())+'.</p>'
content=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Football scoring</title>
<style>
:root{{--ink:#203f30;--muted:#617266;--paper:#f7f6f0;--line:#d8dfd5;--accent:#245b40}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 Arial,sans-serif}}main{{max-width:1080px;margin:auto;padding:42px 40px 65px}}header{{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:20px;font-size:13px}}a{{color:var(--accent);text-underline-offset:4px}}.eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:11px;color:var(--muted)}}h1,h2{{font-family:Georgia,serif;font-weight:normal;line-height:1.12}}h1{{font-size:64px;max-width:800px;margin:30px 0 18px}}h2{{font-size:33px;margin:0 0 20px}}.intro{{max-width:670px;color:var(--muted);font-size:18px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin:38px 0 48px;padding:25px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.number{{font:35px Georgia,serif;display:block}}.label,.small{{font-size:13px;color:var(--muted)}}section{{margin:0 0 42px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:40px}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:normal;text-align:left}}td,th{{padding:12px 9px;border-bottom:1px solid var(--line)}}td:first-child,th:first-child{{padding-left:0}}td:not(:first-child),th:not(:first-child){{text-align:right}}.plot{{margin:20px 0}}.plot svg{{width:100%;height:auto}}.note{{padding:18px 22px;border-left:3px solid #a98039;background:#eeede3;font-size:14px}}.links{{display:flex;gap:24px;flex-wrap:wrap;font-size:14px}}footer{{border-top:1px solid var(--line);padding-top:20px;font-size:12px;color:var(--muted)}}@media(max-width:700px){{main{{padding:24px 20px}}h1{{font-size:43px}}.stats{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr;gap:20px}}table{{font-size:12px}}td,th{{padding:10px 4px}}}}
</style></head><body><main>
<header><span>FOOTBALL · RECRUITMENT SCORING</span><a href="https://github.com/RemiBp/football-recruitment-scoring">Repository ↗</a></header>
<p class="eyebrow" style="margin-top:32px">Group review draft</p><h1>Who reaches ten<br>league goals next season?</h1>
<p class="intro">A benchmark for attacking players with at least 900 prior-season minutes. Nationality is reserved for the fairness audit.</p>
<div class="stats"><div><span class="number">{len(df):,}</span><span class="label">eligible player-seasons</span></div><div><span class="number">{df.outcome_observed.sum():,}</span><span class="label">observed outcomes</span></div><div><span class="number">9</span><span class="label">leagues</span></div><div><span class="number">{run['test_positive_rate']:.1%}</span><span class="label">test players reaching 10 goals</span></div></div>
<section><h2>Information before the season</h2><div class="two"><p>Performance comes from the prior season. Valuations are the latest record strictly before July 1. <strong>{val:,} of {val:,}</strong> dated valuations pass this check; {len(df)-val} are missing.</p><p>Train: 2014–2018 and 2021.<br>Validation: 2022. Test: 2023–2025.<br><span class="small">COVID seasons crossing the cutoff are excluded. No player ID or nationality enters a model.</span></p></div></section>
<section><h2>Out-of-time performance</h2><table><thead><tr><th>Model</th><th>AUC</th><th>95% interval</th><th>Avg. precision</th><th>Brier ↓</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="small">{run['test_rows']:,} test observations. 200 player-cluster bootstrap samples. Intervals describe uncertainty for fitted models; they do not prove a ranking. {foundationnote}</p>{matched}<div class="plot">{svg}</div></section>
<section><h2>Stability, measured separately</h2><table><thead><tr><th>Model</th><th>AUC range, 5 seeds</th><th>Mean score change, noise</th><th>Score PSI</th></tr></thead><tbody>{''.join(stab)}</tbody></table><p class="small">Noise: 1% of training SD on continuous measurements. PSI compares validation scores with test scores. Coefficient intervals, input PSI and 8/15-goal sensitivity are included in the repository.</p><p class="small">Largest input shifts: {html.escape(psitext)}.</p></section>
<section class="two"><div><h2>Fairness is still a question</h2><p>Nationality error rates are reported overall and within positions, with minimum group counts. A valuation ablation is available.</p><p class="small">Differences are descriptive. Position, league, playing time and coverage can affect them. They do not establish a nationality penalty or a causal valuation effect.</p></div><div><h2>What needs group review</h2><p>Confirm the 10-goal threshold and league scope. Finish the group's interpretation and fairness analysis before recommending a model.</p><p class="small">International caps are unavailable. Profile height, foot and citizenship are snapshot metadata. Some valuations are stale.</p></div></section>
<section class="note"><strong>{miss:.1%} of eligible rows have no observed next-season outcome.</strong> They are kept as unknown, not negative. Results apply to continued observation in the selected leagues; exits can also leave partial goal totals.</section>
<div class="links"><a href="https://github.com/RemiBp/football-recruitment-scoring/blob/main/audit.ipynb">Audit notebook</a><a href="https://github.com/RemiBp/football-recruitment-scoring/blob/main/dataset_final.parquet">Dataset</a><a href="https://github.com/RemiBp/football-recruitment-scoring/blob/main/deliverables/Football%20stability.pptx">Two slides</a><a href="https://github.com/RemiBp/football-recruitment-scoring/blob/main/DATA_DICTIONARY.md">Definitions</a></div>
<footer style="margin-top:35px">Source: <a href="https://www.kaggle.com/datasets/davidcariboo/player-scores">Transfermarkt dataset on Kaggle</a>. Sporting performance is a proxy, not a measure of hiring merit. Academic work; no individual recruitment recommendations.</footer>
</main></body></html>'''
# Do not use typographic long dashes in prose; season ranges use plain 'to'.
content=content.replace('2014–2018','2014 to 2018').replace('2023–2025','2023 to 2025')
(out/'index.html').write_text(content)
print(out/'index.html')
