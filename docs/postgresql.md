# PostgreSQL election pipeline

The pipeline uses PostgreSQL for cleaned inputs and SQL feature engineering. Pandas still handles cleaning and passes the resulting training/test datasets to the models. DuckDB is no longer a project dependency.

## Local Codespace setup

With the virtual environment and PostgreSQL 16 installed:

```bash
scripts/mlflow-services.sh init
scripts/election-database.sh
source .mlflow/config.env
source .env.election
jupyter lab
```

If PostgreSQL software is missing, run `scripts/mlflow-services.sh install` first. On subsequent sessions, run `scripts/mlflow-services.sh start`. Python automatically reads the repository's `.env.election` when no explicit database URL or `ELECTION_DATABASE_URL` environment variable is provided, including in an already-running notebook kernel. Source the configuration files for shell commands and MLflow settings.

The election setup script provisions `election_prediction` with its own login, `election_pipeline`, on the existing local server (port 5433). It does not alter MLflow's database or credentials. Its generated `.env.election` file is private and Git-ignored. The election role owns its database and is not a superuser. The setup script does not reset credentials if the role/database already exists.

Stopping the server with `scripts/mlflow-services.sh stop` also makes the election database unavailable. Both databases live in `.mlflow/postgres/`; preserve that directory across container rebuilds. Database state does not survive deletion of the Codespace. MLflow backups alone do not back up the separate election database.

## Another PostgreSQL server

Create a database owned by a dedicated login with permission to create schemas, then configure its connection:

```bash
export ELECTION_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/election_prediction'
```

Use your deployment's TLS settings for remote connections, and URL-encode special characters in credentials. Supply this variable through your shell or secret manager; it overrides the local `.env.election` file. Configure MLflow separately as described in [mlflow.md](mlflow.md).

`apply_sql_queries(cleaned_data, database_url=...)` also accepts an explicit PostgreSQL DSN, which takes precedence over the environment and local file. Without any configuration it fails with setup guidance; it never defaults to the MLflow database.

## What a run stores

Each SQL run creates a unique schema named `run_<UTC timestamp>_<random suffix>`. Inside it:

- `input_*` tables retain the cleaned DataFrames, bulk-loaded using `COPY`.
- Election/polling views normalize inputs without duplicating their stored data; feature views implement the six SQL stages.
- `train_data` and `test_data` expose the prepared datasets.
- `snapshot_info` records completion time, input/output row counts and the SQL used.

All these changes commit in one transaction. SQL failures roll back the entire new schema. Concurrent runs use different schemas. Earlier snapshots are retained, and each model's compact MLflow summary includes `metadata.prepared_data.database_schema`, linking its results to the snapshot. Returned DataFrames also carry that name in `attrs['database_schema']`.

Feature views calculate from the retained inputs when queried; they are not materialized feature tables. Historical snapshots should be treated as read-only by users, although the database owner can modify them. This is run-based snapshotting, rather than incremental updates or a live progress dashboard. The loader logs stages through Python logging; enable INFO logging if you want console progress:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Inspecting data

After sourcing `.env.election`:

```bash
psql "$ELECTION_DATABASE_URL"
```

List schemas with `\dn`. Substitute an actual run schema in these queries:

```sql
SELECT * FROM run_EXAMPLE.snapshot_info;
SELECT COUNT(*) FROM run_EXAMPLE.train_data;
SELECT * FROM run_EXAMPLE.test_data LIMIT 10;
```

Successful runs accumulate storage. Back up useful snapshots and explicitly remove obsolete run schemas when no longer needed. Schema names are recorded in MLflow, so removing one also removes that run's inspectable data.

## Verification

```bash
source .mlflow/config.env
source .env.election
python -m pytest
```

Database tests exercise bulk loading (NULLs, empty strings, quoted text and mixed-case columns), feature engineering, persistent run isolation and transactional rollback. They remove their own committed snapshots and use rollback-only schemas for other checks. Database integration tests skip if `ELECTION_DATABASE_URL` is absent; a configured but unreachable database fails the tests.

PostgreSQL folds unquoted identifiers to lowercase, so mixed-case `natSW` columns and the polling `Date` column are explicitly quoted. Missing numeric values remain missing; they are not filled with zero during loading.

References: [PostgreSQL schemas](https://www.postgresql.org/docs/16/sql-createschema.html), [Psycopg bulk loading](https://www.psycopg.org/docs/cursor.html#cursor.copy_expert).
