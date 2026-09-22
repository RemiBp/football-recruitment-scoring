# Football recruitment scoring

Academic comparison of sporting-performance scores for a hypothetical club. The output estimates whether an attacking player reaches a goals threshold next season. It is not a recommendation to hire or reject a named player. Nationality is retained for aggregate fairness audits, not used by the primary models.

## Decisions fixed before fitting

- Main outcome: at least 10 domestic-league goals in season N. Sensitivity: 8 and 15. These are separate outcomes, not a promise of a 20-25% event rate. The brief's undefined quartile is reported on training outcomes only, not used to choose a convenient class balance.
- Eligible population: attacking positions and at least 900 domestic-league minutes in N-1; adult players at the prediction cutoff.
- July 1 of year N is the information cutoff. Performance features use only dated appearances before that cutoff. Valuation joins are strictly backward, including the year-earlier valuation used for growth.
- The official season in games is authoritative. Month-based season inference is unsafe for COVID extensions and calendar-year leagues.
- Missing next-season appearances mean unknown coverage, not zero goals. Such rows remain in the released table with a missing label and an explicit observation flag. Models condition on continued observation; attrition is reported.
- National-team caps and goals must have historical dates to be usable. Latest profile totals cannot be substituted. Unsupported requested features remain explicitly unavailable.
- Club valuation is reconstructed from the prior-season squad and each member's valuation before the cutoff. It is a historical proxy, not the provider's current club total.
- Player profile attributes (position, citizenship, foot, height) are snapshot metadata. Their historical validity is not established; position-control results are descriptive. Passing timestamp tests does not certify all metadata as historically known.
- Train, validation and test use different target seasons in chronological order. The same player may recur across seasons: this corresponds to forecasting known players. A separate unseen-player test sensitivity is reported. No player identifiers or names enter the models.
- All imputation, scaling, categorical encoding and PSI bins are fit on training data only. The final test is used for evaluation, not for choosing model settings.
- Bootstrap samples are clustered by player to preserve repeated seasons. Training bootstraps refit preprocessing and logistic coefficients; test bootstraps quantify conditional AUC uncertainty. They answer different questions.
- PSI uses training quantile bins, explicit missing values, and smoothing. It is descriptive and is not a significance test.
- Random-seed comparisons hold data fixed. Perturbations are simulated measurement noise, not a causal intervention.

## Model scope

Logistic regression, Random Forest and XGBoost implement the requested stability comparisons. XGBoost is not a tabular foundation model. The course additionally requires a foundation model, so TabICL is included as a fourth comparator where execution succeeds. A missing foundation-model result is reported as incomplete, never substituted by XGBoost.

## Claims to test

The proposed nationality differences, valuation-mediated proxy effects and superior stability of logistic regression are hypotheses. No model winner or discrimination finding is specified in advance. Nationality differences in error rates can reflect position, league, sample size, coverage and other confounding factors. Scoring goals is not itself an objective measure of hiring merit.

## Sources

- Data: https://www.kaggle.com/datasets/davidcariboo/player-scores
- Publisher / snapshot status: https://github.com/dcaribou/transfermarkt-datasets
- Tabular foundation model: https://github.com/soda-inria/tabicl
- Employment-use framing: EU AI Act, Annex III 4(a), https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3 . Exact legal classification depends on intended use and the relevant exceptions, not the project title alone.
- French recruitment discrimination provisions: https://travail-emploi.gouv.fr/discriminations-lembauche-de-quoi-parle-t . No conclusion on a particular club or player is drawn here.

The brief's claim about nationality premiums in valuations still needs specific supporting papers. Do not present it as an established result of this dataset.
