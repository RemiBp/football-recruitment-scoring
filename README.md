# Football recruitment scoring

Course project: can last season's performance help estimate whether an attacking player scores at least 10 league goals next season?

The dataset and code are ready for group review. Cloud model runs are being checked; conclusions will be added from the actual outputs.

## Start here

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
