# Project directories

Paths are relative to the project root.

| Directory | Purpose |
|---|---|
| `docs/` | MLflow setup instructions and example configuration. |
| `scripts/` | Starts local MLflow services and validates experiment tracking. |
| `src/` | Installable Python source code. |
| `src/election/` | Main election prediction package and shared path utilities. |
| `src/election/data/` | Bundled historical election workbook and data notes. |
| `src/election/preparation/` | Cleans historical results, boundary-adjusted results and polling. |
| `src/election/sql/` | DuckDB queries for feature engineering and train/test datasets. |
| `src/election/pipeline/` | Coordinates the full data preparation and modelling workflow. |
| `src/election/pipeline/helpers/` | Supports data loading, cleaning, predictor documentation and model selection. |
| `src/election/models/` | Model definitions, training, evaluation and MLflow tracking. |
| `src/election/models/adapters/` | Gives individual model implementations a common interface. |
| `tests/` | Automated checks for data processing, models and pipeline integration. |
| `notebooks/` | Notebook entry point for running the pipeline. |
| `notebooks/analysis/` | Development notebooks exploring models and features. |
| `notebooks/analysis/01_initial_overview/` | Initial data exploration. |
| `notebooks/analysis/02_baseline_models/` | Baseline model experiments. |
| `notebooks/analysis/03_xgboost/` | Initial XGBoost experiments. |
| `notebooks/analysis/04_logistic_regression_improvement/` | Logistic regression refinements. |
| `notebooks/analysis/05_nn_first_development/` | Reserved for initial neural network development; currently empty. |
| `notebooks/analysis/06_exploring_multilayer_model/` | Multilayer neural network and extended XGBoost experiments. |
| `notebooks/analysis/07_improving_feature_engineering/` | Feature engineering and expanded predictor experiments. |
| `notebooks/notes/` | Detailed project description and proposed improvements. |
| `notebooks/outputs/` | Exported datasets, predictions, predictor descriptions and evaluation results. |

Local tooling and generated directories are grouped below; their internal database, environment and cache folders are managed automatically.

| Directory | Purpose |
|---|---|
| `.git/` | Git version history and repository metadata. |
| `.venv/` | Local Python environment and installed dependencies. |
| `.pytest_cache/` | Cached pytest information. |
| `__pycache__/` (throughout the project) | Cached compiled Python files. |
| `src/election_prediction.egg-info/` | Generated Python package installation metadata. |
| `.mlflow/` | Local MLflow PostgreSQL data, connection socket and saved artifacts. |
| `.mlflow-demo/` | Artifacts from MLflow validation runs. |
| `.agents/`, `.codex/` | Workspace directories reserved for coding agent configuration. |
| `.aws/` | Workspace directory reserved for AWS configuration. |
