# Election experiment tracking in a Codespace

The complete pipeline compares the registry's eight candidates on five outer elections, selects from the current execution's recorded mean accuracies, retunes/refits the winner using pre-2024 history, and evaluates 2024. Only after final outputs and required logging succeed does its parent run finish successfully. No full experiment is launched by setup or validation.

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

This second command runs the full, expensive experiment; use it only when ready. Restart the notebook kernel/server if it was launched before setting the environment. There is no SQLite or file-store fallback. Database/logging failures abort the execution. If the server becomes unreachable, runs can remain RUNNING because failure status could not be written; inspect the exception notes and treat them as incomplete, never successful. No autologging is enabled. Use a fresh kernel without externally enabled autologging.

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

Select the election-prediction experiment in MLflow. Each execution has a parent; its eight candidate children each have five election children. A ninth child is the winner's final-test run. Changing the registry/schedule explicitly changes the expected candidate/election list; default executions retain eight and five.

The parent records expected evaluations, shared split and environment references, Git commit and dirty state, the winner, criterion and tie decision. MLflow supplies run identifiers and start/end timestamps. Candidate artifacts give the adapter and policy references, fixed configuration and relevant neural or two-stage structure. Election and final artifacts contain only actual selected settings, training years and per-seed refit durations. Final runs reference the winning candidate specification.

Only accuracy, changed-seat accuracy, retained-seat accuracy, macro F1 and log loss are persisted as score metrics. Candidate metrics are their equal-election means, with undefined subgroup elections omitted exactly as before; the number contributing to each mean is stored as a parameter. Election and final scores carry evaluation and subgroup counts. Missing previous winners are excluded from changed/retained groups, not from total accuracy. Undefined subgroup scores have an explicit undefined tag and no numeric metric; they are never zero. Macro F1 retains the complete declared party list and zero_division=0. Counts are parameters, so they do not duplicate score metrics across levels.

Comparison and selection query only the current execution's candidate and election runs and require complete, valid records. Exact ties use registry order, and inner configuration ties retain grid order. Final outcomes never enter tuning or selection. Successfully finished children remain available if later work fails. Subsequent executions create separate histories.

The notebook displays MLflow-backed comparisons and the returned final scores; it no longer evaluates 2024 separately. Prepared train/test CSVs, the existing predictor guide, final test_predictions.csv, and test_confusion_matrix.csv/.png remain local pipeline outputs. Historical score CSVs, historical per-seat predictions, exhaustive tuning JSON and redundant final test_metrics.json are no longer generated. Existing saved files are left untouched. There was no separate model-saving implementation to change.

The recorded information supports **recreating the procedure and retraining**, not restoring a fitted model. No learned weights, model objects, checkpoints, inner trials, epoch histories, datasets, source copies or per-observation predictions are uploaded. Source, dependency and predictor references point to the existing project; predictor documentation is not duplicated. Seeds do not guarantee identical learned weights or predictions. A dirty Git working tree must be retained/committed by you if its exact source is needed later; no commit is created automatically.

## Small validation demonstration

After starting services and sourcing config:

```bash
source .mlflow/config.env
.venv/bin/python scripts/validate_mlflow.py
```

The demonstration uses tiny synthetic data and cheap adapters, not election data or the full training grid. Its clearly named validation experiment is separate from production. It checks two executions, the hierarchy, compact records, failure handling, final output availability, and reads the first history/artifact again after restarting services. It leaves validation history available for inspection. Unit/integration tests can be run with pytest; PostgreSQL-backed tests require the same configured server.

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

Restore artifacts to the **original absolute artifact path** recorded in the backed-up database. If recovering into a differently named workspace, recreate that original directory under /workspaces and configure MLFLOW_ARTIFACT_ROOT accordingly before starting services. Do not use an archive from an untrusted source. Verify a known execution and download its selected_configuration.json in the UI before resuming training. Use a matching MLflow version for restore, then perform supported schema upgrades with a fresh backup. pg_dump covers the dedicated database; initialization recreates its one local role, so cluster-wide roles are not a separate required export here.

References: [MLflow tracking server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html), [PostgreSQL pg_dump](https://www.postgresql.org/docs/16/app-pgdump.html), [PostgreSQL pg_restore](https://www.postgresql.org/docs/16/app-pgrestore.html).

## Election pipeline database

MLflow stores experiment metadata in the `mlflow` database. Election inputs and feature snapshots use the separate `election_prediction` database and `ELECTION_DATABASE_URL`. See [PostgreSQL setup](postgresql.md). Source both configuration files before starting Jupyter or running the complete test suite.
