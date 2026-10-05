# Project directories

Paths are relative to the project root.

| Directory | Purpose |
|---|---|
| `setup/` | Sets up PostgreSQL and MLflow. |
| `src/` | Installable Python source code. |
| `src/election/` | Main election prediction package and shared path utilities. |
| `src/election/data/` | Bundled historical election workbook and data notes. |
| `src/election/preparation/` | Cleans historical results, boundary-adjusted results and polling. |
| `src/election/sql/` | PostgreSQL queries for feature engineering and train/test datasets. |
| `src/election/pipeline/` | Coordinates the full data preparation and modelling workflow. |
| `src/election/pipeline/helpers/` | Supports data loading, cleaning and predictor documentation. |
| `src/election/models/` | Shared model support, nested evaluation, comparison, MLflow tracking. |
| `src/election/reports/` | Terminal commands for saved accuracy history, comparisons, evaluation details and prepared-data manifests. |
| `src/election/models/adapters/` | Gives individual model implementations a common interface. |
| `tests/` | Automated checks for data processing, models and pipeline integration, including synthetic MLflow validation. |
| `notebooks/` | Notebook entry point for running the pipeline. |
| `notes/` | Detailed project description, proposed improvements and retained predictor guide. |
| `outputs/` | Latest prepared-snapshot pointer (`latest_prepared.json`), dataset metadata and reports subfolders. |
| `outputs/reports/` | Optional final predictions and confusion-matrix CSV/image reports, overwritten on a final-stage rerun. |
| `outputs/datasets/<snapshot>/` | Prepared-data manifests and generated predictor guides; train/test rows are stored in PostgreSQL. |

## Important files

| File | Purpose |
|---|---|
| `README.md` | Project overview and brief commands for running the pipeline and printing 2024 accuracy history. |
| `pyproject.toml` | Package metadata, dependencies, source discovery and pytest configuration. |
| `setup.cfg` | Places packaging staging files under `/tmp/election-prediction-build`. |
| `setup/mlflow-services.sh` | Installs, initialises and manages the local PostgreSQL and MLflow services. |
| `setup/election-database.sh` | Creates the separate election database and private connection configuration. |
| `.env.election` | Private election database URL; excluded from Git. |
| `.mlflow/config.env` | Generated private service configuration; excluded from Git. |
| `src/election/reports/accuracy_history.py` | Prints completed 2024 MLflow evaluations without training. |
| `tests/test_mlflow_pipeline.py` | Synthetic pipeline integration test formerly in `setup/validate_mlflow.py`. |
| `notes/detailed_project_description.md` | Entire process, setup, independent stages, evaluation and model specifications. |
| `notes/intended_next_improvements.md` | Proposed future changes. |
| `notes/predictor_descriptions.md` | Retained predictor reference guide. |

PostgreSQL retains successful preparation runs in separate schemas containing cleaned inputs, feature views, `train_data`, `test_data` and snapshot metadata. The election and MLflow databases are separate, while the default local PostgreSQL service stores both under `.mlflow/postgres/`. Dataset manifests identify the retained election schema. MLflow stores compact evaluation summaries and metrics.

## Local tooling and generated directories

Local tooling and generated directories are grouped below; their internal database, environment and cache folders are managed automatically.

| Directory | Purpose |
|---|---|
| `.git/` | Git version history and repository metadata. |
| `.venv/` | Local Python environment and installed dependencies. |
| `.pytest_cache/` | Cached pytest information. |
| `__pycache__/` (throughout the project) | Cached compiled Python files. |
| `src/election_prediction.egg-info/` | Generated Python package installation metadata. |
| `.mlflow/` | Local MLflow PostgreSQL data, connection socket and saved artifacts. |
| `.agents/`, `.codex/` | Workspace directories reserved for coding agent configuration. |
| `.aws/` | Workspace directory reserved for AWS configuration. |

The generated guide in each snapshot documents that dataset’s columns; the notes guide is a reading reference. Keep `latest_prepared.json` and snapshot metadata for independent stages. Reports are optional and can be regenerated. Model accuracy history is stored in MLflow, and train/test rows are stored in PostgreSQL.
