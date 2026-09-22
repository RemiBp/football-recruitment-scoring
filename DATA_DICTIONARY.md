# Table contract

One row per `(player_id, target_season)`. `schema.json` defines the exact ordered columns and types. Schema 1.0.0 is a candidate for the group's freeze; changes after approval require rerunning every analysis.

## Population and outcome

Adult players whose minute-weighted modal position in prior-season lineups is Centre-Forward, Left Winger, Right Winger, Second Striker or Attacking Midfield; at least 900 league minutes in that season.

Nine leagues with long coverage and a comparable single-phase schedule: England, France, Germany, Italy, Spain, Netherlands, Portugal, Russia and Türkiye. Cups, European matches and international matches do not count towards the domestic-goals outcome. Newly added leagues and leagues with separate playoff phases are excluded. League coverage is recorded in `reports/league_coverage.csv`; date and game-count checks are quality screens, not proof of exhaustive coverage.

`target_10` is 1 if observed goals in the next season are at least 10, otherwise 0. `target_8` and `target_15` provide sensitivity outcomes. No appearance in the selected leagues means a missing outcome. An exit partway through a season can also leave a partial observed total; this remains a limitation.

- Train target seasons: 2014, 2015, 2016, 2017, 2018, 2021.
- Validation target season: 2022.
- Test target seasons: 2023, 2024, 2025.
- COVID target seasons 2019 and 2020 are excluded: their full prior or target seasons do not fit the July 1 boundary.

## Model inputs

| Column | Definition at the prediction cutoff |
|---|---|
| `goals_prev`, `assists_prev` | Sum over prior-season appearances in the selected leagues |
| `minutes_prev`, `appearances_prev` | Prior-season minutes and unique games |
| `goals_per90_prev` | `90 * goals_prev / minutes_prev` |
| `age` | Days from birth to July 1, divided by 365.25 |
| `position` | Minute-weighted modal recorded prior-season lineup position |
| `league` | Prior-season league with the largest player-club minutes |
| `valuation_eur` | Last dated valuation strictly before July 1 |
| `valuation_growth_12m` | Ratio to last valuation strictly before the previous July 1, minus 1; missing for unavailable or zero baseline |
| `club_mean_valuation_eur` | Mean available as-of valuations of the prior-season club squad; not a current club total |
| `height_cm`, `foot` | Current profile metadata; historical provenance not established |
| `club_change_known_at_cutoff` | Observed transfer after last prior-season appearance and before July 1 to a different club; ambiguous same-day moves are missing |

Monetary inputs use `log1p`, numerical missing values use training medians, and numerical features use training standard deviations. Categories are one-hot encoded from training data; unseen categories are ignored. The same preprocessing convention applies to all primary models.

## Audit and unavailable fields

`citizenship_audit` is current profile citizenship, retained only for aggregate audits. It is not ethnicity and should not be used as a substitute for ethnicity. Citizenship changes and multiple citizenships are not reconstructed.

`observed_international_appearances` and `observed_international_goals` are entirely missing in this snapshot: no national-team appearances join. They remain in the contract as explicitly unavailable fields and are excluded from the 14 model inputs. Current career totals from player profiles are not a valid substitute.

`player_id`, club IDs, target-season numbers, all outcome fields and all audit timestamps are excluded from model inputs. Timestamps make the information boundary auditable. Missing valuations cannot pass as observed values; the audit reports both dated coverage and strict-date validity.

`club_valued_players / club_squad_players` describes coverage of the club valuation proxy. The squad is formed from observed prior-season players and can include players who leave the club.

## Interpretation limits

This is a forecast conditional on continued observation, not an estimate for every eligible player. Missing outcomes vary by league. Nationality-stratified errors can reflect position, league, playing time, movement and coverage. Position-stratified tables help describe composition; they do not identify a causal effect or prove discrimination. The 0.5 error-rate threshold is a fixed descriptive convention, not a recruitment policy.
