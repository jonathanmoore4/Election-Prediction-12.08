# Can UK general election results be predicted without using constituency identities?

Constituency identifiers can improve predictive accuracy but may reduce a model's ability to generalise across future boundary reviews and changing electoral landscapes. This project investigates whether the outcome of the 2024 UK general election can instead be predicted using only features that remain meaningful across elections, including the previous winning party, previous winning party’s vote share, national polling, the incumbent government, and the constituency's country or region.

The project combines historical election results and national polling in an end-to-end Python and SQL pipeline. Candidate classification models are compared using a temporal validation election before the selected model is refitted and evaluated on the 2024 General Election. The candidates include logistic regression, random forest, XGBoost and three neural-network variants.

The previous XGBoost model achieved **69.94% accuracy** on the 2024 election. The latest saved neural-network evaluation achieves **58.94%**, despite outperforming XGBoost on the held-out 2019 validation election. Neural-network development now compares architectures and learning-rate policies across multiple historical elections to improve generalisation; the evaluation section below explains the different samples and the status of these results.

## Motivation

Going into the 2024 election, a substantial Labour victory was widely expected, representing a markedly different national political environment from recent elections.

Models that rely heavily on constituency identity or historical constituency patterns may struggle to generalise when political conditions or constituency boundaries change. This project therefore investigates how much constituency-level behaviour can be captured using publicly available information that can be applied consistently between elections.

Alongside the research question, the project has been developed as a reproducible machine-learning workflow in which data ingestion, cleaning, feature engineering, model selection and evaluation can be coordinated through a single pipeline.

## Quick start

Use Python 3.11 or newer and install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Open [00_run_pipeline.IPYNB](Analysis%20and%20model%20development/00_pipeline/00_run_pipeline.IPYNB) and run all cells. Start Jupyter from the repository root or a directory within it. The pipeline downloads the configured sources, checks the supplied local workbook, prepares features, compares models on 2019 and retrains the selected model. The notebook then evaluates it on complete 2024 rows and displays a confusion matrix.

Training includes XGBoost grid search and neural-network ensembles, so a full run can take time. Internet access is required for ingestion. Most dependency versions are not pinned, and remote inputs can change.

## Outputs

Train and test CSVs are saved only in `TEST_TRAIN/`. Model accuracies are saved alongside the pipeline notebook:

```text
TEST_TRAIN/
├── train.csv
├── test.csv
└── predictor_descriptions.md
Analysis and model development/00_pipeline/
├── 00_run_pipeline.IPYNB
└── model_accuracies.csv
```

The CSVs and `TEST_TRAIN/predictor_descriptions.md` are created or overwritten when the notebook runs. The guide describes every exported column, distinguishes predictors from outcomes and metadata, and documents units and missing values. `model_accuracies.csv` records each candidate's initial 2019 accuracy, changed-seat accuracy, evaluation row counts and combined selection score. These are validation results before retraining, not 2024 test scores. Accuracies are stored as fractions.

The fitted model remains in memory; the confusion matrix and printed test results appear in the notebook. No model file or separate plot file is currently exported. Save the notebook to retain its displayed outputs.

For Python callers, `run_pipeline(output_dir=...)` controls the model accuracy output directory and returns `[fitted_model, model_name]`. Train and test always go to the project's `TEST_TRAIN/` directory. Without an argument, model accuracies also go to `TEST_TRAIN/`. Relative score output paths are resolved against the caller's working directory.

## Repository guide

| Location | Contents |
|---|---|
| `Run Pipeline/run_pipeline.py` | End-to-end workflow and CSV output handling |
| `Run Pipeline/additional_funcs/` | Source loading, cleaning coordination, model selection and tests |
| `Current Data Preparation/` | Election, polling and boundary-change cleaning functions |
| `SQL/` | Five ordered feature-building queries, executed in an in-memory DuckDB database |
| `Models/` | Logistic regression, random forest, XGBoost and three neural-network variants |
| `Analysis and model development/00_pipeline/` | Pipeline notebook and its generated CSVs |
| `Analysis and model development/01_initial_overview/` | Initial data overview notebook |
| `Analysis and model development/02_baseline_models/` | Baseline comparison notebook |
| `Analysis and model development/03_XGboost/` | XGBoost analysis notebook |
| `Analysis and model development/04_logistic_regression_improvement/` | Logistic-regression diagnostics notebook |
| `Analysis and model development/05_nn_first_development/` | Neural-network study, input snapshot, results, reports and reproduction scripts |
| `data/manual/` | Checksum-verified local workbook and provenance notes |
| `TEST_TRAIN/` | Existing data exports and default output location for direct Python calls |

## Data and predictors

Each row represents a Great Britain constituency at an election; Northern Ireland is excluded. The existing training export contains 5,705 rows across nine elections from 1987 to 2019. The existing test export contains 632 rows for 2024. These are saved-data counts, and completeness filters reduce the rows used for evaluation.

The target groups winning parties into Conservative (`con`), Labour (`lab`), Liberal Democrat (`lib`), SNP/Plaid Cymru (`natSW`) and Other (`oth`). Constituency identifiers support joins and reporting but are not predictors; election year controls data splitting rather than serving as a predictor.

The shared predictors include country/region, previous winning party and its last-election vote share, previous party vote shares, Conservative/Labour/Liberal Democrat national polling (`con_polling`, `lab_polling`, `lib_polling`), the governing party and three projected party shares.

The SQL currently calculates each projected share as:

```text
previous constituency vote share + current national polling − previous national seat share
```

The last term is a share of constituency winners, not a national vote share. Reviewing this feature definition remains a useful development task.

## Model comparison

Candidates are trained using elections before 2019 and evaluated on the same complete 2019 rows, including all target classes. A changed seat is one whose winner differs from its known previous winner.

Selection maximises **overall accuracy + changed-seat accuracy**, giving the two metrics equal weight. If no changed seats are available, selection uses overall accuracy. Exact ties follow candidate order: XGBoost, random forest, logistic regression, NN01, NN02, NN03. The winner is passed the full pre-2024 training dataframe for its model-specific retraining procedure.

| Candidate | Training procedure |
|---|---|
| XGBoost | Five-fold stratified grid search over tree count and depth; handles missing numeric values |
| Random forest | Median-imputed numeric features; fixed random seed |
| Logistic regression | Standardised numeric features; excludes incomplete rows and rows with `oth` as winner or previous winner during training |
| NN01 | Hidden layers of 32 and 16 units; four candidate learning rates |
| NN02 | Hidden layers of 64 and 32 units; eight candidate learning rates |
| NN03 | One hidden layer of 16 units; fixed learning rate of 0.3 |

Each neural variant uses ten seeds, holds out the latest supplied whole election for early stopping, and averages checkpoint probabilities. Rate selection uses ensemble validation log loss. Retraining repeats this procedure, retaining a whole-election validation holdout rather than fitting every supplied row.

The [neural-network development package](Analysis%20and%20model%20development/05_nn_first_development/README.md) contains the historical study and reproduction commands, with evaluation windows restricted to 1997–2019.

## Evaluation status and limitations

The latest saved neural-network run outperforms XGBoost on the **held-out 2019 validation election**: overall accuracy is **90.30% versus 87.76%** on the same 629 complete rows. On the 75 seats whose winning party changed, the neural network achieves **34.67% versus 4.00%**. Its higher combined score leads the pipeline to select it for retraining.

That validation advantage does not carry through to the **2024 test election**. The selected neural network achieves **58.94% on 621 of 632 seats**, excluding 11 rows with missing predictors or winners. The previous XGBoost model's reported 2024 result was **69.94% across all 632 seats**. The neural network therefore underperforms the earlier model in the recorded test results, although the different evaluation samples mean this is not a controlled comparison on identical rows. Stronger performance on one held-out election has not translated into stronger performance on the next.

Development is addressing this through whole-election validation, early stopping, ten-seed probability ensembles, and comparisons of network size and learning-rate policy across seven historical evaluation elections from 1997 to 2019. NN02 expands the hidden layers to 64/32 and searches eight rates; NN03 tests a smaller 16-unit network with a fixed rate of 0.3. The historical study favours different configurations for overall accuracy and changed-seat performance, so these variants are candidates for improvement rather than established solutions to the 2024 shortfall.

The pipeline notebook's retained output comes from the earlier four-candidate run, before NN02 and NN03 were added. It does not establish the 2024 performance of the current six-candidate pipeline. Rerun all cells to obtain updated validation scores and test evaluation.

Accuracy measures correctly classified constituency winners, not vote-share error or national seat-total error. The notebook reports excluded rows and displays a confusion matrix. Further work could add per-party precision/recall, changed-seat test metrics and seat-total errors.

Model-family selection uses one validation election. XGBoost's internal folds mix elections, while neural networks hold out an entire election. Constituencies within an election share national conditions, so thousands of rows do not represent thousands of independent electoral environments. Historical development comparisons and repeated inspection of 2024 results also limit claims of untouched out-of-sample performance.

## Focused checks

From the repository root:

```bash
python -m unittest discover -s "Run Pipeline/additional_funcs" -p "test_*.py"
python -m unittest discover -s Models -p "test_*.py"
```

These cover source loading, model selection and neural-network training/retraining behaviour. They do not replace a full pipeline run against the remote data sources.

## Data Sources and Acknowledgement

| Dataset | Source | Purpose |
|---|---|---|
| Historical general election results | House of Commons Library | Constituency results and historical vote shares |
| 2024 General Election results | House of Commons Library | Final evaluation |
| 2005 and 2019 notional results | UK Parliament | Previous-election information across boundary changes |
| 1992 notional results | Rallings and Thrasher BBC Media Guide | Previous-election information across boundary changes |
| Historical national polling | Mark Pack's PollBase | Pre-election Conservative, Labour and Liberal Democrat polling |
| Scottish boundary changes | Electoral Calculus | Approximate mapping across the 2005 Scottish boundary changes |

### Second-party vote shares

The election cleaners calculate `second_party_vote_share` as the second-highest available share among the same party categories used for `winning_party_vote_share`. Ties occupy separate ranking positions. Historical and 1992 notional results include the aggregate other-party category; the 2005 and 2019 notional cleaners use their existing four main-party categories, while 2024 uses its broader party list. This therefore measures the second-ranked available category, which may differ from an individual runner-up candidate. The 2001 notional data retains the historical share when remapping constituencies.

Fewer than two non-missing shares give a missing second share. The 1992 cleaner retains its existing convention of treating missing votes and zero-total shares as zero. SQL carries the feature into the final datasets and joins the matched actual or notional comparison as `previous_second_party_last_election_vote_share`; an unmatched previous result remains missing. Regenerate the train/test CSVs through the pipeline to populate these new columns. Current-election shares are outcome information; only the previous-election feature is suitable as a pre-election predictor. Existing model feature lists are unchanged.

SQL step `5_further_feature_engineering.sql` adds `previous_margin_1st_2nd` as `previous_winning_party_last_election_vote_share - previous_second_party_last_election_vote_share`. The margin uses the same proportion scale as the shares and remains missing if either share is missing. Step `6_reading_into_testtrain.sql` then splits the enriched data into training elections and the 2024 test election.
