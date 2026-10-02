# Can UK general election results be predicted without using constituency identities?

Constituency identifiers can improve predictive accuracy but may reduce a model's ability to generalise across boundary reviews and changing electoral landscapes. This project investigates whether constituency winners can instead be predicted using information that remains meaningful across elections: previous party vote shares, national polling, the governing party and country or region.

The project combines historical election results and polling in an end-to-end Python and SQL pipeline. The current iteration compares **eight modelling procedures across five historical elections**, tuning each procedure using only earlier elections. The selected procedure is then tuned again on pre-2024 history and evaluated retrospectively on the 2024 General Election.

The saved complete comparison selects **Conditional XGBoost**, with **88.85% mean historical accuracy** across 2005, 2010, 2015, 2017 and 2019. The saved pipeline notebook reports **75.00% accuracy on all 632 Great Britain constituencies in 2024** (474 correct predictions). On the same constituencies, predicting the previous winner achieves **52.37%** (331 correct), predicting Labour everywhere achieves **65.03%** (411 correct), and predicting Conservative everywhere achieves **19.15%** (121 correct).  These are recorded results; rerunning with different inputs or dependencies may change them.

## Motivation

The 2024 election represented a markedly different political environment from recent elections. A model that performs well on one historical election may struggle when national conditions change. This project tests how much constituency behaviour can be captured without using constituency identity as a predictor, while developing a reproducible workflow for ingestion, cleaning, feature engineering, tuning and evaluation.

## Quick start

Use Python 3.11 or newer and install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks,dev]"
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Open [00_run_pipeline.ipynb](notebooks/00_run_pipeline.ipynb) and run all cells. Start Jupyter from the repository root or a directory within it. The notebook downloads the configured sources, checks the supplied local workbook, prepares features, compares all candidates and fits the selected procedure for 2024. It then predicts every test row, scores rows with known winners and displays a confusion matrix.

A full run includes nested searches and ten-network ensembles and can take substantial time. Internet access is required for ingestion. Most dependency versions are not pinned, and remote inputs can change.

## Package and development

The project follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/#creating-the-package-files) with a `src/` layout and regular packages marked by `__init__.py`. `pyproject.toml` defines the build backend, project metadata and dependencies. Core installation uses `python -m pip install -e .`, so the pipeline runs directly from `src/`; the `notebooks` extra adds notebook/report tools and `dev` adds testing tools.

Import project modules explicitly so calls show their origin:

```python
from election.pipeline import run_pipeline

selection = run_pipeline.run_pipeline(output_dir="notebooks/outputs")
```

Run tests with `python -m pytest`. Normal development and pipeline execution do not require building a distribution. If packaging is needed, `setup.cfg` directs setuptools staging files to `/tmp/election-prediction-build` instead of creating a `build/` directory in this checkout. SQL queries and the checksum-verified local workbook are included in the installed package. When installed outside a checkout, default outputs go under the working directory's `notebooks/outputs/`; use `output_dir` to choose another location.

Python modules and project-owned files use lowercase snake case; package directories use lowercase names. Standard filenames such as `README.md` and `__init__.py` retain their conventional spelling. Analysis notebooks remain separate from the installable package.

## Outputs

| Location | Output |
|---|---|
| `notebooks/outputs/train.csv`, `test.csv` | Prepared historical training data and 2024 test data |
| `notebooks/outputs/predictor_descriptions.md` | Guide to exported columns, units and missing values |
| `notebooks/outputs/model_accuracies.csv` | Candidate scorecard across five historical elections |
| `notebooks/outputs/model_accuracies_elections.csv` | Metrics and selected settings for each candidate and election |
| `notebooks/outputs/model_accuracies_predictions.csv` | Historical constituency predictions and party probabilities |
| `notebooks/outputs/model_accuracies_report.json` | Tuning records, final-fit details and run status |
| `notebooks/outputs/test_predictions.csv` | Final test rows with predicted winners, exported by the notebook |
| `notebooks/outputs/test_metrics.json` | Final test accuracy and evaluation row counts, exported by the notebook |
| `notebooks/outputs/test_confusion_matrix.csv`, `test_confusion_matrix.png` | Final confusion matrix as data and an image, exported by the notebook |

All exports default to `notebooks/outputs/`. `run_pipeline(output_dir=...)` redirects every export to the specified directory; relative paths resolve from the caller's working directory. Reruns overwrite data exports and reports. CSV accuracies are fractions. The fitted model remains in memory.

## Repository guide

| Directory | Purpose |
|---|---|
| `notebooks/` | Notebook for running the complete pipeline |
| `src/election/pipeline/` | Coordinates source loading, data preparation, exports and model selection |
| `src/election/preparation/` | Cleans election, polling and boundary-change data |
| `src/election/sql/` | Builds predictors and separates training and test data in DuckDB |
| `src/election/models/` | Model candidates, preprocessing, training and historical evaluation |
| `analysis/` | Development notebooks and experiments, retained for reference; this directory is not used to run the project |
| `src/election/data/` | Packaged local source workbook |
| `notebooks/outputs/` | All generated datasets, predictor guide and evaluation reports |
| `tests/` | Automated checks for the pipeline and models |

## Data and predictors

Each row represents a Great Britain constituency at an election; Northern Ireland is excluded. The existing exports contain 5,705 training rows across nine elections from 1987 to 2019 and 632 test rows for 2024. These are saved-data counts and may change when inputs are regenerated.

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

Missing predictors are imputed using training data within each fold. Numeric values use medians (zero for entirely missing columns); categories use a missing marker, with a historical fallback for previous winners. Exported CSVs retain missing values.

## Historical tuning and model selection

The shared evaluator implements **nested walk-forward validation**: tuning happens inside each historical forecast, and both levels keep whole elections together.

| Outer election scored | Earlier elections used for inner validation | History used for outer refit |
|---|---|---|
| 2005 | 1997, 2001 | 1987–2001 |
| 2010 | 1997, 2001, 2005 | 1987–2005 |
| 2015 | 1997, 2001, 2005, 2010 | 1987–2010 |
| 2017 | 1997, 2001, 2005, 2010, 2015 | 1987–2015 |
| 2019 | 1997, 2001, 2005, 2010, 2015, 2017 | 1987–2017 |

For each configuration and inner election, fit a fresh model using only earlier elections. Select settings by mean inner-election accuracy, giving elections equal weight. Refit a fresh model on all history before the outer election and save its predictions. Outer outcomes do not determine preprocessing, settings or training duration.

Candidate selection uses **mean accuracy across the five outer elections**, with equal election weights. Changed-seat accuracy is a diagnostic and receives no extra selection weight. Exact ties follow registry order; configuration ties follow configuration order.

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

Each neural candidate averages probabilities from ten networks with predefined seeds. Inner fits select each seed's checkpoint by validation log loss, while learning rates are ranked by mean inner-election accuracy. For the outer and final refits, each seed trains on all available history for its median best-checkpoint duration from the selected configuration's inner folds, rounded halves up with a minimum of one epoch. These refits do not hold back another election or monitor forecast outcomes.

## Data Sources and Acknowledgement

| Dataset | Source | Purpose |
|---|---|---|
| Historical general election results | House of Commons Library | Constituency results and historical vote shares |
| 2024 General Election results | House of Commons Library | Final evaluation |
| 2005 and 2019 notional results | UK Parliament | Previous-election information across boundary changes |
| 1992 notional results | Rallings and Thrasher BBC Media Guide | Previous-election information across boundary changes |
| Historical national polling | Mark Pack's PollBase | Pre-election Conservative, Labour and Liberal Democrat polling |
| Scottish boundary changes | Electoral Calculus | Approximate mapping across the 2005 Scottish boundary changes |
