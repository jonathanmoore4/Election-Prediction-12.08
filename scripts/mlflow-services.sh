#!/usr/bin/env bash
# Single-user Codespace services; state survives container rebuilds under /workspaces.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE_ROOT=${ELECTION_MLFLOW_HOME:-$REPO_ROOT/.mlflow}
case "$STATE_ROOT" in /workspaces/*) ;; *) echo 'ELECTION_MLFLOW_HOME must be an absolute path under /workspaces' >&2; exit 1;; esac
CONFIG_FILE=$STATE_ROOT/config.env
COMMAND=${1:-status}
if [[ "$COMMAND" == install ]]; then
    sudo apt-get update
    sudo apt-get install -y postgresql-16 postgresql-client-16
    "$REPO_ROOT/.venv/bin/python" -m pip install -e "$REPO_ROOT[dev,notebooks]"
    exit
fi
mkdir -p "$STATE_ROOT"
chmod 700 "$STATE_ROOT"
if [[ ! -f "$CONFIG_FILE" ]]; then
    if [[ "$COMMAND" != init ]]; then
        echo 'First run: scripts/mlflow-services.sh init' >&2; exit 1
    fi
    STATE_ROOT="$STATE_ROOT" "$REPO_ROOT/.venv/bin/python" - <<'PY'
import os, secrets, shlex
from pathlib import Path
root = Path(os.environ['STATE_ROOT'])
password = secrets.token_urlsafe(32)
settings = dict(PGDATA=str(root/'postgres'), PGPORT='5433', PGHOST='127.0.0.1',
                PGUSER='mlflow', PGDATABASE='mlflow', PGPASSWORD=password,
                MLFLOW_BACKEND_STORE_URI=f'postgresql+psycopg2://mlflow:{password}@127.0.0.1:5433/mlflow',
                MLFLOW_TRACKING_URI='http://127.0.0.1:5000',
                MLFLOW_EXPERIMENT_NAME='election-prediction',
                MLFLOW_ARTIFACT_ROOT=str(root/'artifacts'))
path = root/'config.env'
path.write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k,v in settings.items()))
path.chmod(0o600)
PY
fi
# Configuration is private, generated locally, and never checked into Git.
source "$CONFIG_FILE"
export PATH="/usr/lib/postgresql/16/bin:$PATH"
mkdir -p "$STATE_ROOT/socket" "$MLFLOW_ARTIFACT_ROOT"
PG_CTL=$(command -v pg_ctl || true)
if [[ -z "$PG_CTL" ]]; then
    echo 'PostgreSQL 16 software missing; run scripts/mlflow-services.sh install, then reuse this state.' >&2; exit 1
fi
if [[ -f "$PGDATA/PG_VERSION" && $(cat "$PGDATA/PG_VERSION") != 16 ]]; then
    echo 'Existing PostgreSQL major version differs; install matching binaries or use documented dump/restore upgrade.' >&2; exit 1
fi
start_postgres() {
    if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
        pg_ctl -D "$PGDATA" -l "$STATE_ROOT/postgres.log" -o "-p $PGPORT -h 127.0.0.1 -k $STATE_ROOT/socket" -w start
    fi
}
stop_mlflow() {
    if [[ -f "$STATE_ROOT/mlflow.pid" ]]; then
        pid=$(cat "$STATE_ROOT/mlflow.pid")
        if kill -0 "$pid" 2>/dev/null; then
            # Verify the recorded process before signalling; do not kill a reused PID.
            if ps -p "$pid" -o args= | grep -q 'mlflow server'; then
                kill "$pid"
                for _ in {1..50}; do
                    kill -0 "$pid" 2>/dev/null || break
                    sleep .2
                done
                if kill -0 "$pid" 2>/dev/null && [[ $(ps -p "$pid" -o stat=) != Z* ]]; then
                    echo 'MLflow is still stopping; retry status/stop.' >&2; exit 1
                fi
            fi
        fi
        rm "$STATE_ROOT/mlflow.pid"
    fi
}
start_mlflow() {
    if [[ -f "$STATE_ROOT/mlflow.pid" ]] && kill -0 "$(cat "$STATE_ROOT/mlflow.pid")" 2>/dev/null; then
        if ps -p "$(cat "$STATE_ROOT/mlflow.pid")" -o args= | grep -q 'mlflow server'; then
            echo 'MLflow already running'; return
        fi
    fi
    # Environment variable supplies backend credentials without putting them in argv.
    "$REPO_ROOT/.venv/bin/mlflow" db upgrade "$MLFLOW_BACKEND_STORE_URI" >"$STATE_ROOT/migration.log" 2>&1
    export MLFLOW_DISABLE_AGENT_HINT=1
    export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
    allowed_hosts='localhost:*,127.0.0.1:*'
    allowed_origins='http://localhost:*,http://127.0.0.1:*'
    if [[ -n ${CODESPACE_NAME:-} && -n ${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-} ]]; then
        forwarded_host="${CODESPACE_NAME}-5000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
        allowed_hosts="$allowed_hosts,$forwarded_host"
        allowed_origins="$allowed_origins,https://$forwarded_host"
    fi
    setsid nohup "$REPO_ROOT/.venv/bin/mlflow" server --host 127.0.0.1 --port 5000 --workers 1 \
        --no-serve-artifacts --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
        --allowed-hosts "$allowed_hosts" --cors-allowed-origins "$allowed_origins" </dev/null >"$STATE_ROOT/mlflow.log" 2>&1 &
    echo $! >"$STATE_ROOT/mlflow.pid"
    for _ in {1..60}; do
        if curl --fail --silent "$MLFLOW_TRACKING_URI/health" >/dev/null; then
            echo "MLflow ready at $MLFLOW_TRACKING_URI (forward port 5000 privately)"; return
        fi
        if ! kill -0 "$(cat "$STATE_ROOT/mlflow.pid")" 2>/dev/null; then
            echo "MLflow failed; inspect $STATE_ROOT/mlflow.log" >&2; exit 1
        fi
        sleep .5
    done
    echo "MLflow startup timed out; inspect $STATE_ROOT/mlflow.log" >&2; exit 1
}
case "$COMMAND" in
    init)
        if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
            initdb -D "$PGDATA" -U "$PGUSER" --auth-local=peer --auth-host=scram-sha-256 \
                --pwfile=<(printf '%s\n' "$PGPASSWORD") >/dev/null
        fi
        start_postgres
        if [[ $(psql -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname='mlflow'") != 1 ]]; then
            createdb mlflow
        fi
        start_mlflow
        ;;
    start) start_postgres; start_mlflow ;;
    stop) stop_mlflow; if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then pg_ctl -D "$PGDATA" -m fast -w stop; fi ;;
    restart) "$0" stop; "$0" start ;;
    status)
        pg_ctl -D "$PGDATA" status
        curl --fail --silent "$MLFLOW_TRACKING_URI/health"
        ;;
    *) echo 'Usage: scripts/mlflow-services.sh {install|init|start|stop|restart|status}' >&2; exit 1 ;;
esac
