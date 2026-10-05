# Project directories

Paths are relative to the project root.

| Directory | Purpose |
|---|---|
| `setup/` | Sets up PostgreSQL and MLflow, validates tracking and provides example configuration. |
| `src/` | Installable Python source code. |
| `src/election/` | Main election prediction package and shared path utilities. |
| `src/election/data/` | Bundled historical election workbook and data notes. |
| `src/election/preparation/` | Cleans historical results, boundary-adjusted results and polling. |
| `src/election/sql/` | PostgreSQL queries for feature engineering and train/test datasets. |
| `src/election/pipeline/` | Coordinates the full data preparation and modelling workflow. |
| `src/election/pipeline/helpers/` | Supports data loading, cleaning and predictor documentation. |
| `src/election/models/` | Model definitions, training, evaluation and MLflow tracking. |
| `src/election/models/adapters/` | Gives individual model implementations a common interface. |
| `tests/` | Automated checks for data processing, models and pipeline integration. |
| `notebooks/` | Notebook entry point for running the pipeline. |
| `notes/` | Detailed project description, proposed improvements and retained predictor guide. |
| `outputs/` | Exported datasets, predictions, predictor descriptions and evaluation results. |

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
