# Election experiment tracking in a Codespace

Preparation, historical evaluation, comparison and final 2024 evaluation run
independently. The full pipeline connects those same functions. MLflow retains
one compact run per completed model evaluation, not a hierarchy of execution,
candidate and election runs. No full experiment is launched by setup or validation.

## Install and start

Run these commands in the repository's GitHub Codespace. The existing virtual environment is used. If absent, create it first with Python 3.11 or newer:

```bash
python -m venv .venv
scripts/mlflow-services.sh install
scripts/mlflow-services.sh init
source .mlflow/config.env
```

The install command installs PostgreSQL 16 and the project's Python dependencies. The init command generates a private password, initializes a dedicated PostgreSQL cluster owned by your Codespace user on localhost port 5433, creates the mlflow database, applies MLflow migrations and starts the server on localhost port 5000. It safely reuses existing data. The distribution's default PostgreSQL cluster is not used. Credentials are generated in the ignored .mlflow/config.env with mode 600; the enclosing directory has mode 700. Do not print or commit that file. A secret-free example is in docs/mlflow.env.example.

In the Codespaces Ports tab, forward port 5000, set its visibility to **Private**, and open it in your browser. Port 5433 needs no forwarding. The server allows localhost and the current Codespace's forwarded hostname. Keep the browser address separate from MLFLOW_TRACKING_URI: training inside the Codespace connects to http://127.0.0.1:5000.

Run the notebook notebooks/00_run_pipeline.ipynb from an environment that has sourced the config, or call the entry point:

```bash
source .mlflow/config.env
.venv/bin/python -c 'from election.pipeline.run_pipeline import run_pipeline; run_pipeline()'
```

This second command runs the full, expensive experiment; use it only when ready. Restart the notebook kernel/server if it was launched before setting the environment. There is no SQLite or file-store fallback. Database/logging failures abort the execution. If the server becomes unreachable, runs can remain RUNNING because failure status could not be written; inspect the exception notes and treat them as incomplete, never successful. No autologging is enabled. Tracked evaluation explicitly disables autologging, including in a reused notebook kernel.

## Everyday services and persistent storage

```bash
scripts/mlflow-services.sh status
scripts/mlflow-services.sh restart
scripts/mlflow-services.sh stop
scripts/mlflow-services.sh start
source .mlflow/config.env
```

PostgreSQL data is in /workspaces/Election-Prediction-12.08/.mlflow/postgres; compact JSON artifacts are in the separate .mlflow/artifacts directory. Logs and config are alongside them. These are Git-ignored locations under /workspaces, which survive ordinary Codespace stops, restarts and container rebuilds. They do **not** survive deletion of the Codespace. Software installed in the container may need reinstalling after a rebuild: rerun install, then start, without deleting .mlflow. Use the same PostgreSQL major version (16) for that existing cluster; upgrading majors requires PostgreSQL's supported upgrade/dump-restore procedure. Recreate the virtual environment after a rebuild if its interpreter no longer matches.

To choose another persistent directory before first initialization, set ELECTION_MLFLOW_HOME to an absolute path under /workspaces and use that same value on every service command. Add any alternative repository-local state directory to .gitignore yourself. Avoid moving an existing artifact directory: existing MLflow runs contain its absolute location. Artifacts are accessed directly by the client and server inside the same Codespace; they are not proxied or uploaded to another service. The server uses one request worker and disables optional MLflow background job execution; summary tracking needs no scoring or tracing jobs. The service script is intended for one user, run sequentially.

## Viewing the retained records

Select the `election-prediction` experiment in MLflow. A historical model run has
`purpose=model_comparison`, five `accuracy_YEAR` metrics and `mean_outer_accuracy`.
A final run has `purpose=final_evaluation`, `accuracy_2024` and
`mean_outer_accuracy` (the same score because this schedule has one outer election).
Eight model comparisons followed by one final evaluation create nine runs.

Each run has one `evaluation.json`. It contains the candidate grid, full schedule,
selected parameters and winning inner-election scores for each outer election,
training years and per-seed neural refit durations. Metadata includes model and
architecture identifiers, fixed training settings and protocol, feature columns,
target/scoring definitions, data identity and prepared-data/schema references.
MLflow supplies completion time and run identity. Final metadata references the
historical selection run and, for the full pipeline, all comparison source runs.

There are no child runs, candidate-fit records, epoch histories, package inventories,
dataset rows, fitted weights or constituency predictions in MLflow. Explicit
logging is used; autologging and system metrics are not enabled by this project.
Tracked evaluation explicitly disables autologging, including in a reused notebook kernel.

All required elections must have valid results before recording starts. The run is
marked complete only after its compact artifact and metrics are written, and
logging failures abort the calling stage. A partially written or failed run cannot
enter comparison. If termination could not be written because the server became
unreachable, treat RUNNING records as incomplete.

`compare_recorded` reads the preparation manifest and MLflow, without loading data
rows or importing model implementations. It selects the most recently completed
compatible historical run per requested model by MLflow end time, then ranks these
runs by mean outer accuracy. Compatibility requires identical data identity, full
schedule, target, scoring and evaluation protocol. Exact model ties follow the
requested model order; configuration ties preserve ParameterGrid order. Missing
or excluded records are returned with reasons. The full pipeline refuses to select
from a partial set of requested models. Final evaluations never enter comparison.

The neural procedure is unchanged: inner scoring elections select checkpoints;
outer/final refits train on all eligible history for each seed's median best inner
epoch, rounded halves up, without validation monitoring. Outer/final outcomes do
not enter checkpoint selection. The protocol is documented in
`src/election/models/adapters/nn01_model.py` and recorded in the summary.

Prepared datasets are retained in content-identified
`notebooks/outputs/datasets/<snapshot>/` folders. The preparation manifest records
train/test locations and hashes; a later preparation does not overwrite a previous
snapshot. `latest_prepared.json` points to the latest snapshot. A hash does not
preserve the dataset: keep the corresponding files and/or retained SQL schema.
Final prediction CSVs and confusion matrices remain local optional outputs.

Existing hierarchical runs and old local outputs are not deleted or relabelled.
They remain inspectable, but the new comparison and result reader only use
`record_type=evaluation-v1` compact runs. Rerun the desired historical models once
to populate the new comparison format. No migration is needed for tracking storage.

## Independent stages

```bash
python -m election.pipeline.run_pipeline prepare
python -m election.pipeline.run_pipeline evaluate --model logistic_regression
python -m election.pipeline.run_pipeline evaluate --model all
python -m election.pipeline.run_pipeline compare
python -m election.pipeline.run_pipeline final --selection-run HISTORICAL_RUN_ID
python -m election.pipeline.run_pipeline all --data-dir notebooks/outputs --reuse-evaluations
```

For Python usage and optional output/data directories, see [README](../README.md).
The final stage accepts a completed historical run, verifies its compatibility,
reuses its candidate grid and fixed settings, tunes again through 2019 and scores
2024. It returns a dictionary, not a fitted model.

## Small validation demonstration

After starting services and sourcing config:

```bash
source .mlflow/config.env
.venv/bin/python scripts/validate_mlflow.py
```

This uses tiny synthetic data and inexpensive functions in a separate, uniquely
named validation experiment. It checks the independent preparation/evaluation/
comparison/final pipeline wiring, nine compact runs per execution, latest-compatible
selection, final local outputs and reading retained results with a new client.
It does not train real election models or restart the user's services.

Unit tests use small synthetic datasets and reduced fitting budgets. Real MLflow
integration checks require `MLFLOW_TRACKING_URI`; PostgreSQL data checks also
require `ELECTION_DATABASE_URL`. Source both configuration files before pytest to
include those checks. The recorded settings support retraining the procedure,
not restoring fitted weights; seeds alone do not guarantee identical predictions.

## Backup both stores

Stop training first, stop the MLflow server to prevent writes, and leave PostgreSQL running for a consistent logical dump. Do not simply copy a running PostgreSQL data directory. These commands use PostgreSQL's supported custom-format pg_dump and separately archive artifacts:

```bash
source .mlflow/config.env
export PATH=/usr/lib/postgresql/16/bin:$PATH
# Stop both services, then start only PostgreSQL for backup.
scripts/mlflow-services.sh stop
pg_ctl -D "$PGDATA" -l .mlflow/postgres.log -o "-p $PGPORT -h 127.0.0.1 -k $PWD/.mlflow/socket" -w start
mkdir -p .mlflow/backups
backup_stamp=$(date -u +%Y%m%dT%H%M%SZ)
pg_dump --format=custom --file=".mlflow/backups/$backup_stamp.dump" "$PGDATABASE"
tar -czf ".mlflow/backups/$backup_stamp-artifacts.tar.gz" -C "$MLFLOW_ARTIFACT_ROOT" .
scripts/mlflow-services.sh start
```

Copy **both** files outside the Codespace (for example download using the Files panel). Store the private config separately in secure storage if needed, or generate new credentials on restore. Credentials are not included in the artifact archive. Dumps may contain experiment metadata and references. Both backups must correspond to the same paused execution state. Adapt paths if using ELECTION_MLFLOW_HOME.

## Restore

Restore into a fresh dedicated tracking database, with no pipeline or server writing. Install matching software and run init if recovering in a new Codespace; this establishes the role and fresh credentials. Then stop services and start PostgreSQL alone as above. Confirm the target before using dropdb: this intentionally replaces the destination history.

```bash
source .mlflow/config.env
export PATH=/usr/lib/postgresql/16/bin:$PATH
scripts/mlflow-services.sh stop
pg_ctl -D "$PGDATA" -l .mlflow/postgres.log -o "-p $PGPORT -h 127.0.0.1 -k $PWD/.mlflow/socket" -w start
# Substitute the actual downloaded backup filenames.
dropdb "$PGDATABASE"
createdb "$PGDATABASE"
pg_restore --exit-on-error --no-owner --no-privileges --dbname="$PGDATABASE" /path/to/backup.dump
# Use an empty artifact destination; move any existing directory aside first.
mv "$MLFLOW_ARTIFACT_ROOT" "$MLFLOW_ARTIFACT_ROOT.before-restore"
mkdir -p "$MLFLOW_ARTIFACT_ROOT"
tar -xzf /path/to/backup-artifacts.tar.gz -C "$MLFLOW_ARTIFACT_ROOT"
scripts/mlflow-services.sh start
```

Restore artifacts to the **original absolute artifact path** recorded in the backed-up database. If recovering into a differently named workspace, recreate that original directory under /workspaces and configure MLFLOW_ARTIFACT_ROOT accordingly before starting services. Do not use an archive from an untrusted source. Verify a known execution and download its evaluation.json (or selected_configuration.json for an old hierarchical run) in the UI before resuming training. Use a matching MLflow version for restore, then perform supported schema upgrades with a fresh backup. pg_dump covers the dedicated database; initialization recreates its one local role, so cluster-wide roles are not a separate required export here.

References: [MLflow tracking server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html), [PostgreSQL pg_dump](https://www.postgresql.org/docs/16/app-pgdump.html), [PostgreSQL pg_restore](https://www.postgresql.org/docs/16/app-pgrestore.html).

## Election pipeline database

MLflow stores experiment metadata in the `mlflow` database. Election inputs and feature snapshots use the separate `election_prediction` database and `ELECTION_DATABASE_URL`. See [PostgreSQL setup](postgresql.md). Source both configuration files before starting Jupyter or running the complete test suite.
