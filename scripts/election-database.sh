#!/usr/bin/env bash
# Provision the pipeline database in the existing local PostgreSQL cluster.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE_ROOT=${ELECTION_MLFLOW_HOME:-$REPO_ROOT/.mlflow}
if [[ ! -f "$STATE_ROOT/config.env" ]]; then
    echo 'First initialize PostgreSQL with scripts/mlflow-services.sh init' >&2
    exit 1
fi
source "$STATE_ROOT/config.env"
export PATH="/usr/lib/postgresql/16/bin:$PATH"
if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
    pg_ctl -D "$PGDATA" -l "$STATE_ROOT/postgres.log" \
        -o "-p $PGPORT -h 127.0.0.1 -k $STATE_ROOT/socket" -w start
fi
REPO_ROOT="$REPO_ROOT" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import secrets
import shlex
from urllib.parse import quote
import psycopg2
from psycopg2 import sql

config = Path(os.environ['REPO_ROOT']) / '.env.election'
if config.exists():
    print('Pipeline configuration already exists; source .env.election to use it.')
    raise SystemExit(0)
connection = psycopg2.connect(dbname='postgres')
connection.autocommit = True
try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = 'election_pipeline'")
        role_exists = cursor.fetchone() is not None
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'election_prediction'")
        database_exists = cursor.fetchone() is not None
        if role_exists or database_exists:
            raise SystemExit('Election role/database already exists. Supply its existing DSN in .env.election; credentials will not be reset.')
        password = secrets.token_urlsafe(32)
        cursor.execute(sql.SQL('CREATE ROLE election_pipeline LOGIN PASSWORD {}').format(sql.Literal(password)))
        cursor.execute('CREATE DATABASE election_prediction OWNER election_pipeline')
    url = f"postgresql://election_pipeline:{quote(password, safe='')}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/election_prediction"
    # Create the private file without a window where credentials are world-readable.
    descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write('export ELECTION_DATABASE_URL=' + shlex.quote(url) + '\n')
finally:
    connection.close()
print('Election database ready. Run: source .env.election')
PY
