# Football recruitment scoring

Course project: can last season's performance help estimate whether an attacking player scores at least 10 league goals next season?

The dataset, four model comparisons and 200-replicate stability results are ready for group review. [Cloud run](https://github.com/RemiBp/football-recruitment-scoring/actions/runs/35740388579): all jobs passed. This is a working contribution, not the complete group submission.

## Results

Test AUC: Logistic 0.805, Random Forest 0.792, XGBoost 0.802, TabICL 0.811. TabICL uses a 1,500-row training context; matched classical baselines are provided. The paired bootstrap interval for Logistic minus XGBoost spans zero. Logistic scores move least under the specified measurement-noise test. None of these results establishes nationality discrimination.

## Start here

- `docs/index.html`: concise standalone report. Download it and open locally.
- `deliverables/Football stability.pptx`: two editable slides, with speaker notes. [Preview 1](deliverables/previews/slide-1.png) · [Preview 2](deliverables/previews/slide-2.png)
- `deliverables/Speaker notes.md`: 90-second script and Q&A.
- `audit.ipynb`: executed audit notebook.
- `HANDOFF.md`: group message and decisions to confirm.

- `dataset_final.parquet`: 8,428 player-seasons across nine leagues; 6,823 observed outcomes.
- `build_dataset.py`: raw CSVs to a reproducible, typed table.
- `schema.json`: frozen column order, types and model features, version 1.0.0.
- `PROTOCOL.md`: population, information cutoff, exclusions and claims to test.
- `stability.py`: Logistic, Random Forest, XGBoost; 200 player-cluster bootstraps, PSI, seeds, noise and target sensitivity.
- `foundation.py`: TabICL, with a disclosed 1,500-row training context and matched classical baselines.

Nationality is audit-only. Individual player scores are not published. Missing next-season coverage is not treated as zero goals. The table is a review candidate, not confirmation that the group or professor has approved the topic.

## Reproduce

Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python stability.py --bootstraps 200
# Optional: regenerate the table from the source snapshot (about 234 MB compressed).
python build_dataset.py --download
# Foundation-model comparator:
pip install -r requirements-foundation.txt
python foundation.py
```

The GitHub Actions **Reproducible analysis** workflow runs the checks, stability work and foundation comparator on cloud runners. Use **Run workflow** to reproduce both model jobs. Results are downloadable artifacts; they do not silently replace reviewed results in the repository.

## Source and boundaries

[Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores), publisher snapshot 6 July 2026. Raw files stay outside Git; `data/manifest.json` records their hashes.

The outcome is a sporting-performance proxy, not hiring merit. Differences between nationality groups are not proof of discrimination. Better stability of logistic regression is a hypothesis, not a required conclusion. Historical coverage, transfers out of the selected leagues and snapshot player metadata limit what the study can establish.
