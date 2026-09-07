# Can UK general election results be predicted without using constituency identities?

Constituency identifiers can improve predictive accuracy but may reduce a model's ability to generalise across future boundary reviews and changing electoral landscapes. This project investigates whether the outcome of the 2024 UK general election can instead be predicted using only features that remain meaningful across elections, including the previous winning party, previous majority proportion, national polling, the incumbent government, and the constituency's country or region.

The project combines historical election results and national polling in an end-to-end Python and SQL pipeline. Candidate classification models are compared using a temporal validation election before the selected model is refitted and evaluated on the 2024 General Election.

The current pipeline records an accuracy of **55.06%** on the 2024 test data.

## Motivation

Going into the 2024 election, a substantial Labour victory was widely expected, representing a markedly different national political environment from recent elections.

Models that rely heavily on constituency identity or historical constituency patterns may struggle to generalise when political conditions or constituency boundaries change. This project therefore investigates how much constituency-level behaviour can be captured using publicly available information that can be applied consistently between elections.

Alongside the research question, the project has been developed as a reproducible machine-learning workflow in which data ingestion, cleaning, feature engineering, model selection and evaluation can be coordinated through a single pipeline.

## Repository structure

**Data Preparation/**
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

## Installation and usage

Install the required dependencies with:

`pip install -r requirements.txt`

Then start Jupyter from the project root and run:

`Analysis and model development/04_first_pipeline.IPYNB`

The notebook currently calls `run_pipeline()`, which downloads and prepares the data, applies the SQL transformations, trains and selects a model, and returns the fitted model for evaluation against the 2024 results.

Dependencies are not currently pinned to exact versions, and the pipeline relies on external data sources, so future reruns are not guaranteed to reproduce identical results.

# Current Analysis — 07/09/2026

## Data and scope

Each observation represents a constituency at a general election. The project covers Great Britain (England, Scotland and Wales), with Northern Ireland excluded.

Historical results from 1997 onwards are used during data preparation. Elections from 2001 to 2019 form the modelling data, while 2024 is used for final evaluation.

The target is the winning party, grouped as Conservative, Labour, Liberal Democrat, Nationalist (SNP and Plaid Cymru), or Other. Constituency names and identifiers are retained for joining and reporting but are not model predictors.

## Pipeline

The project has developed from separate exploratory cleaning and modelling scripts into a reusable end-to-end workflow.

### 1. Data ingestion

`Run Pipeline/additional_funcs/read_in_raw.py` downloads the election, polling, notional-result and boundary-change data from their configured sources.

### 2. Data cleaning

Reusable functions in `Data Preparation/` clean the individual datasets, including adjustments to previous-election information where constituency boundaries have changed.

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

The recorded evaluation covers all 632 Great Britain constituencies in the 2024 test data and reports:

**2024 test accuracy: 55.06%**

For context, Labour won 411 of the 632 constituencies in the test data. Retrospectively predicting Labour for every seat would therefore give an accuracy of **65.03%**. This is a descriptive benchmark based on the realised 2024 labels rather than a forecasting strategy selected in advance, but it highlights the limited performance of the current model.

The result demonstrates that the end-to-end pipeline can produce and evaluate a model on unseen election data, but does not yet establish useful forecasting performance.

## Limitations and future development

Although the modelling dataset contains thousands of constituency observations, it represents only six training elections. Constituencies within an election share the same national political conditions, meaning the number of rows overstates the variety of electoral environments available for training.

Model-family selection currently relies on a single validation election, while XGBoost's internal cross-validation mixes observations from different elections. The evaluation process therefore remains an important area for development.

Further improvements include:

- Expanding evaluation beyond accuracy to metrics that better reflect performance across parties and changed seats, including precision, recall and F1.
- Improving the model-selection and validation procedure, including investigating rolling or grouped validation across elections.
- Further checking predictors for multicollinearity.
- Reviewing the logistic regression model and checking its assumptions, including the relationship between continuous predictors and the log-odds.
- Adding a neural network as an additional candidate model.
- Comparing performance with appropriate baseline models.
- Reviewing the projected-share feature construction.
- Examining predicted national seat totals and confusion patterns.
- Improving the treatment of constituency boundary changes and smaller parties.

## Data Sources and Acknowledgement

| Dataset | Source | Purpose |
|---|---|---|
| Historical general election results | House of Commons Library | Constituency results and historical vote shares |
| 2024 General Election results | House of Commons Library | Final evaluation |
| 2005 and 2019 notional results | UK Parliament | Previous-election information across boundary changes |
| Historical national polling | Mark Pack's PollBase | Pre-election Conservative, Labour and Liberal Democrat polling |
| Scottish boundary changes | Electoral Calculus | Approximate mapping across the 2005 Scottish boundary changes |
