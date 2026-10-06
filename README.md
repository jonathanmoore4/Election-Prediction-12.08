# Can UK general election results be predicted without using constituency identities?

Constituency identifiers can improve predictive accuracy but may reduce a model's ability to generalise across boundary reviews and changing electoral landscapes. This project investigates whether constituency winners can instead be predicted using information that remains meaningful across elections: previous party vote shares, national polling, the governing party and country or region.

The project combines historical election results and polling in an end-to-end Python and PostgreSQL pipeline. MLflow records model configurations, historical accuracy and separate 2024 evaluations so completed runs can be compared and inspected over time. The current iteration compares eight modelling procedures across five historical elections, tuning each procedure using only earlier elections. The selected procedure is then tuned again on pre-2024 history and evaluated retrospectively on the 2024 General Election.

The saved complete comparison selects Conditional XGBoost, with 88.85% mean historical accuracy across 2005, 2010, 2015, 2017 and 2019. The saved pipeline notebook reports 75.00% accuracy on all 632 Great Britain constituencies in 2024 (474 correct predictions). On the same constituencies, predicting the previous winner achieves 52.37% (331 correct), predicting Labour everywhere achieves 65.03% (411 correct), and predicting Conservative everywhere achieves 19.15% (121 correct). These are recorded results; rerunning with different inputs or dependencies may change them.

## View experiment results

[Open the MLflow experiment dashboard](https://jupyter-server-4jr4jr77q4xvc97w-5000.app.github.dev/) to explore recorded model runs, accuracy metrics and configurations. No local setup is required.

This temporary demo is available while the Codespace and MLflow server are running. I am only able to keep this open for short amounts of time.

## Motivation

The 2024 election represented a markedly different political environment from recent elections. A model that performs well on one historical election may struggle when national conditions change. This project tests how much constituency behaviour can be captured without using constituency identity as a predictor, while developing a reproducible workflow for cleaning, feature engineering, hyperparameter tuning and evaluation.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. From the repository root in an already configured Linux Codespace:

```bash
uv sync --extra notebooks
setup/mlflow-services.sh postgres
source setup/environment.sh
uv run mlflow server
```

MLflow runs in the foreground; use Ctrl+C to stop it. If the previous background
server is running, first run `setup/mlflow-services.sh stop-mlflow`. Forward port
5000 privately in Codespaces to open the UI. In a second Bash terminal:

```bash
source setup/environment.sh
uv run python -m election.pipeline.run_pipeline all
```

For a standalone script, use `uv run file_x.py` (replace the filename with an
actual script). Existing runnable package files can also be executed directly,
for example `uv run src/election/reports/accuracy_history.py`. Files that only
define functions need a caller or executable entry point.

`uv.lock` records dependency versions and `.python-version` selects Python 3.14.2.
uv manages `.venv` and installs this package in editable mode; activation is optional.
Run `uv add PACKAGE` to add a runtime dependency, `uv add --dev PACKAGE` for a
development tool, and `uv run pytest` for tests. Notebook tools are optional:
use `uv run --extra notebooks jupyter lab`. PostgreSQL is a separate system
service and is not installed or started by uv. The existing combined background
startup remains available as `setup/mlflow-services.sh start`.

See [first-time setup](notes/detailed_project_description.md#quick-start) before running this on a new checkout. A full run can take substantial time.

To print saved 2024 accuracy history in the shell, with the MLflow configuration sourced:

```bash
uv run python -m election.reports.accuracy_history
```

The report displays a pandas DataFrame with one row per run: accuracy as a percentage, saved model architecture, training protocol, fixed settings, hyperparameter search space, selected 2024 hyperparameters and early stopping settings where applicable. Long cells are shortened for terminal display; add `--full` to show their complete contents. In Python, `accuracy_history(MlflowClient(...))` from `election.reports.accuracy_history` returns the DataFrame with full configuration dictionaries and numeric accuracy proportions. This reads saved results without training; full runs evaluate only the historically selected model on 2024.

## Data and predictors

Each row represents a Great Britain constituency at an election; Northern Ireland is excluded. The saved PostgreSQL datasets contain 5,705 training rows across nine elections from 1987 to 2019 and 632 test rows for 2024. These are saved-data counts and may change when inputs are regenerated.

The target groups winners into Conservative (`con`), Labour (`lab`), Liberal Democrat (`lib`), SNP/Plaid Cymru (`natSW`) and Other (`oth`). All models expose probabilities in this common class order. Constituency identifiers support joins and reporting but are not predictors. Election year controls chronological splitting and missing-data handling; it is metadata rather than an estimator predictor.

The table below groups party-specific columns using `{party}`. `con`, `lab` and `lib` denote Conservative, Labour and Liberal/Liberal Democrat; `natSW` combines SNP/Plaid Cymru and `oth` groups other parties. Shares and polling are proportions: 0.05 represents five percentage points.

For party $p$, let $s_{p,t-1}$ be its previous constituency vote share, $N_{p,t-1}$ its previous national vote share, and $P_{p,t}$ its current pre-election national polling. Previous constituency results use matched actual or notional results where boundaries change; national shares always use the previous actual election.

| Predictor | Meaning or engineering equation |
|---|---|
| `country/region` | Constituency country or English region |
| `previous_winner` | Party with the largest share in the matched previous constituency result |
| `previous_winning_party_last_election_vote_share` | Previous winner's share: $\max_p s_{p,t-1}$ |
| `previous_{party}_share` (`con`, `lab`, `lib`, `natSW`) | $s_{p,t-1} = \text{previous constituency votes for }p / \text{previous constituency valid votes}$ |
| `{party}_polling` (`con`, `lab`, `lib`) | $P_{p,t}$: national polling from the source observation immediately before the election marker |
| `incumbent` | Party in national government before the election |
| `projected_{party}_share` (`con`, `lab`, `lib`) | $s_{p,t-1} + (P_{p,t} - N_{p,t-1})$: previous local share adjusted by national change |
| `previous_{party}_national_vote_share` (all five categories) | $N_{p,t-1} = \sum_c \text{previous votes for }p\text{ in constituency }c / \sum_c \text{previous valid votes in }c$, over retained Great Britain constituencies |
| `supported_incumbent` | $1$ when the previous constituency winner equals the national governing party; $0$ otherwise |
| `incumbent_polling` | $P_{g,t}$ for governing party $g$: Labour polling when Labour governs, otherwise Conservative polling |
| `opposition_polling` | Polling for the other party in the Conservative/Labour pair |
| `{party}_national_change` (all five categories) | $P_{p,t} - N_{p,t-1}$; missing for `natSW` and `oth` because neither has a polling input |
| `holder_polling`, `challenger_polling` | $P_{h,t}$ and $P_{r,t}$ for previous constituency winner $h$ and matched challenger $r$ |
| `holder_national_swing`, `challenger_national_swing` | $P_{h,t} - N_{h,t-1}$ and $P_{r,t} - N_{r,t-1}$ |

Holder/challenger features are missing where their party has no polling input or cannot be matched. Projected shares are not clipped to 0–1. Models select their own feature subsets; the conditional model rearranges party features into previous-winner and challenger roles. Current-election outcomes and constituency identifiers are excluded from predictors.

### Imputation

Missing predictors are imputed using training data within each fold. Numeric values use medians (zero for entirely missing columns); categories use a missing marker, with a historical fallback for previous winners. Prepared PostgreSQL tables retain missing values as SQL NULL.

Further information:

The linked Markdown files below are working notes, updated as the project develops.

- [Project structure](notes/directories.md): directories and important configuration files.
- [Detailed project description](notes/detailed_project_description.md): greater detail about the entire process, from setup and preparation through nested validation, selection, final evaluation and MLflow recording.
- [Intended next improvements](notes/intended_next_improvements.md): proposed new changes and development priorities.

## Historical tuning and model selection

The shared evaluator implements **nested walk-forward validation**: tuning happens inside each historical forecast, and both levels keep whole elections together.

| Outer election scored | Earlier elections used for inner validation | History used for outer refit |
|---|---|---|
| 2005 | 1997, 2001 | 1987–2001 |
| 2010 | 1997, 2001, 2005 | 1987–2005 |
| 2015 | 1997, 2001, 2005, 2010 | 1987–2010 |
| 2017 | 1997, 2001, 2005, 2010, 2015 | 1987–2015 |
| 2019 | 1997, 2001, 2005, 2010, 2015, 2017 | 1987–2017 |

For each configuration and inner election, fit a fresh model using only earlier elections. Select settings by mean inner-election accuracy, giving elections equal weight. Refit a fresh model on all history before the outer election and retain its accuracy. Outer outcomes do not determine preprocessing, settings or training duration.

Candidate selection uses **mean accuracy across the five outer elections**, with equal election weights. Changed-seat accuracy is a diagnostic and receives no extra selection weight. Exact model ties follow requested model order; configuration ties follow configuration order.

After selection, tune the winning procedure again using inner elections from 1997 through 2019, then fit a fresh model on pre-2024 history. Historical fitted models and settings are not averaged into the final model.

### Eight candidates

| Candidate | Configuration search and fitting |
|---|---|
| Logistic regression | L2 regularisation; search `C` = 0.01, 0.1, 1, 10; scaled numeric inputs |
| Random forest | 300 trees; 18 combinations of depth, minimum leaf size and feature sampling; seed 42 |
| XGBoost | Baseline features; 16 combinations of 20/35/50/100 trees and depths 2–5; learning rate 0.05 |
| XGBoost Expanded | Expanded features; independently tunes the same boosted-tree search |
| Conditional XGBoost | Change and challenger classifiers; jointly compares 256 pairs of stage settings using complete winner accuracy |
| NN01 | Hidden layers 32/16; rates 0.1, 0.2, 0.3, 0.5 |
| NN02 | Hidden layers 64/32; rates 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 |
| NN03 | One 16-unit hidden layer; fixed rate 0.3; inner fits still determine refit durations |

Conditional XGBoost assigns the previous winner `P(no change)` and each challenger `P(change) × P(challenger wins | change)`. Its challenger stage trains on changed seats. Both stages form one candidate and their combined winner predictions determine tuning scores.

Each neural candidate averages probabilities from ten networks with predefined seeds. Early stopping is unchanged: each inner scoring election also selects checkpoints, while outer and final outcomes never influence checkpoint selection. Inner fits select each seed's checkpoint by validation log loss, while learning rates are ranked by mean inner-election accuracy. For the outer and final refits, each seed trains on all available history for its median best-checkpoint duration from the selected configuration's inner folds, rounded halves up with a minimum of one epoch. These refits do not hold back another election or monitor forecast outcomes.

## Data Sources and Acknowledgement

| Dataset | Source | Purpose |
|---|---|---|
| Historical general election results | House of Commons Library | Constituency results and historical vote shares |
| 2024 General Election results | House of Commons Library | Final evaluation |
| 2005 and 2019 notional results | UK Parliament | Previous-election information across boundary changes |
| 1992 notional results | Rallings and Thrasher BBC Media Guide | Previous-election information across boundary changes |
| Historical national polling | Mark Pack's PollBase | Pre-election Conservative, Labour and Liberal Democrat polling |
| Scottish boundary changes | Electoral Calculus | Approximate mapping across the 2005 Scottish boundary changes |
