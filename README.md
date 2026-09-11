# Can UK general election results be predicted without using constituency identities?

Constituency identifiers can improve predictive accuracy but may reduce a model's ability to generalise across future boundary reviews and changing electoral landscapes. This project investigates whether the outcome of the 2024 UK general election can instead be predicted using only features that remain meaningful across elections, including the previous winning party, previous majority proportion, national polling, the incumbent government, and the constituency's country or region.

The project combines historical election results and national polling in an end-to-end Python and SQL pipeline. Candidate classification models are compared using a temporal validation election before the selected model is refitted and evaluated on the 2024 General Election.

The latest saved evaluation in [the pipeline notebook](Analysis%20and%20model%20development/04_first_pipeline.IPYNB) records **69.94% accuracy** across all **632 Great Britain constituencies** in the 2024 test data, with no rows excluded. This is an increase of **14.88 percentage points** from the previously reported **55.06%**.

## Motivation

Going into the 2024 election, a substantial Labour victory was widely expected, representing a markedly different national political environment from recent elections.

Models that rely heavily on constituency identity or historical constituency patterns may struggle to generalise when political conditions or constituency boundaries change. This project therefore investigates how much constituency-level behaviour can be captured using publicly available information that can be applied consistently between elections.

Alongside the research question, the project has been developed as a reproducible machine-learning workflow in which data ingestion, cleaning, feature engineering, model selection and evaluation can be coordinated through a single pipeline.

## Repository structure

**Current Data Preparation/**
- Functions for cleaning election, polling and boundary-change data.

**SQL/**
- SQL transformations for combining data and constructing model features.

**Models/**
- Logistic regression, random forest and XGBoost training functions.

**Run Pipeline/**
- `run_pipeline.py` coordinates the end-to-end workflow.
- `additional_funcs/` contains data ingestion, cleaning coordination and automated model selection.

**Analysis and model development/**
- Exploratory analysis, baseline modelling, model development and pipeline evaluation.
- Notebook 04 records the 2024 pipeline evaluation; notebook 05 explores polling-to-seat relationships and constrained 2019 predictions as separate development work.

## Installation and usage

Install the required dependencies with:

`pip install -r requirements.txt`

Then start Jupyter from the project root and run:

`Analysis and model development/04_first_pipeline.IPYNB`

The notebook currently calls `run_pipeline()`, which downloads and prepares the data, applies the SQL transformations, trains and selects a model, and returns the fitted model for evaluation against the 2024 results. Running it rewrites the training and test CSVs in TEST_TRAIN/.

Dependencies are not currently pinned to exact versions, and the pipeline relies on external data sources, so future reruns are not guaranteed to reproduce identical results.

### Manually supplied data

The pipeline also loads the supplied 1997 workbook from [data/manual/results97.xls](data/manual/results97.xls).
Keep this file in Git with the code so a fresh checkout has the same input. Its
SHA-256 checksum is checked before reading, and its path is resolved from the
project root, so it works when running notebooks from another directory.
Install the updated requirements to include the legacy Excel reader, xlrd.

After importing the existing loader, `raw_data = read_raw_data()` makes the
workbook available as `raw_data["results_1997_local"]`. For local-only analysis
without downloading anything, use
`read_raw_data({"results_1997_local": RAW_SOURCES["results_1997_local"]})`,
importing both `read_raw_data` and `RAW_SOURCES` from `read_in_raw`.
The first worksheet is loaded with its title/header rows preserved. Other sheets
can be selected with the source's `kwargs["sheet_name"]` setting.

This adds reproducible access to the workbook; it does not yet incorporate its
contents into model features. See [the provenance record](data/manual/README.md)
for replacement instructions. The original root-level file is retained as supplied.

## Current analysis — 11 September 2026

## Data and scope

Each observation represents a constituency at a general election. The project covers Great Britain (England, Scotland and Wales), with Northern Ireland excluded.

Historical results from 1997 onwards are used during data preparation. Elections from 2001 to 2019 form the modelling data, while 2024 is used for final evaluation.

The target is the winning party, grouped as Conservative, Labour, Liberal Democrat, Nationalist (SNP and Plaid Cymru), or Other. Constituency names and identifiers are retained for joining and reporting but are not model predictors.

## Pipeline

The project has developed from separate exploratory cleaning and modelling scripts into a reusable end-to-end workflow.

### 1. Data ingestion

`Run Pipeline/additional_funcs/read_in_raw.py` downloads the election, polling, notional-result and boundary-change data from their configured sources and reads the checksum-verified local workbook.

### 2. Data cleaning

Reusable functions in Current Data Preparation/ clean the individual datasets, including adjustments to previous-election information where constituency boundaries have changed.

### 3. SQL feature construction

Five ordered SQL queries combine the cleaned datasets, join previous-election information and national polling, construct additional features, and separate the modelling and test data.

### 4. Model training and selection

Logistic regression, random forest and XGBoost classifiers are implemented as reusable training functions.

Models are initially trained using elections before 2019 and compared on their accuracy in predicting the 2019 election. The selected model family is then refitted using the complete training data.

XGBoost additionally tunes the number and maximum depth of trees using five-fold stratified cross-validation. Numeric preprocessing differs between models: logistic regression standardises numeric predictors, random forest median-imputes missing numeric values, and XGBoost handles missing numeric values internally.

## Features

The predictors capture four main sources of electoral information:

- **Geography:** country or region.
- **Previous constituency result:** previous winning party, previous majority proportion and party vote shares.
- **National conditions:** Conservative, Labour and Liberal Democrat polling and the governing party.
- **Engineered features:** projected Conservative, Labour and Liberal Democrat constituency shares.

The projected shares attempt to combine previous constituency-level support with changes in the national electoral environment:

**Projected share = previous constituency vote share + current national polling − previous national seat share**

## Evaluation

The model is currently evaluated in `Analysis and model development/04_first_pipeline.IPYNB`.

The recorded evaluation achieves **69.94% accuracy** across all **632 Great Britain constituencies** in the 2024 test data, with no rows excluded.

Accuracy measures the proportion of constituency winners correctly classified; it does not measure vote-share accuracy or national seat-total error. These figures come from the notebook's saved output, rather than a new pipeline run.

For context, Labour won 411 of the 632 constituencies in the test data. Retrospectively predicting Labour for every seat would therefore give an accuracy of **65.03%**. This is a descriptive benchmark based on the realised 2024 labels rather than a forecasting strategy selected in advance, and the current model exceeds it by **4.91 percentage points**.

One evaluation election does not establish reliable performance across future elections. The notebook also displays a confusion matrix to examine errors by party. It does not record the selected model family or a per-party precision, recall and F1 report, so the headline result should be attributed to the selected pipeline model rather than to a named classifier.

## Limitations and future development

Although the modelling dataset contains thousands of constituency observations, it represents only nine training elections. Constituencies within an election share the same national political conditions, meaning the number of rows overstates the variety of electoral environments available for training.

Model-family selection currently relies on a single validation election, while XGBoost's internal cross-validation mixes observations from different elections. The evaluation process therefore remains an important area for development.

Further improvements include:

- Expanding evaluation beyond accuracy to metrics that better reflect performance across parties and changed seats, including precision, recall and F1.
- Improving the model-selection and validation procedure, including investigating rolling or grouped validation across elections.
- Further checking predictors for multicollinearity.
- Reviewing the logistic regression model and checking its assumptions, including the relationship between continuous predictors and the log-odds.
- Adding a neural network as an additional candidate model.
- Comparing performance with appropriate baseline models.
- Reviewing the projected-share feature construction.
- Quantifying national seat-total errors and extending the notebook's confusion-matrix analysis.
- Improving the treatment of constituency boundary changes and smaller parties.

## Data Sources and Acknowledgement

| Dataset | Source | Purpose |
|---|---|---|
| Historical general election results | House of Commons Library | Constituency results and historical vote shares |
| 2024 General Election results | House of Commons Library | Final evaluation |
| 2005 and 2019 notional results | UK Parliament | Previous-election information across boundary changes |
| 1992 notional results | Rallings and Thrasher BBC Media Guide | Previous-election information across boundary changes |
| Historical national polling | Mark Pack's PollBase | Pre-election Conservative, Labour and Liberal Democrat polling |
| Scottish boundary changes | Electoral Calculus | Approximate mapping across the 2005 Scottish boundary changes |
