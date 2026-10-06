# Source this file once per Bash terminal: source setup/environment.sh
# Keep credentials in the existing private configuration, outside Git.
_ELECTION_ENV_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_ELECTION_ENV_STATE=${ELECTION_MLFLOW_HOME:-$_ELECTION_ENV_ROOT/.mlflow}
if [[ ! -f "$_ELECTION_ENV_STATE/config.env" ]]; then
    echo 'First run: setup/mlflow-services.sh init' >&2
    return 1
fi
source "$_ELECTION_ENV_STATE/config.env"
if [[ -f "$_ELECTION_ENV_ROOT/.env.election" ]]; then
    source "$_ELECTION_ENV_ROOT/.env.election"
fi
export MLFLOW_DEFAULT_ARTIFACT_ROOT="$MLFLOW_ARTIFACT_ROOT"
export MLFLOW_SERVE_ARTIFACTS=false
export MLFLOW_HOST=127.0.0.1
export MLFLOW_PORT=5000
export MLFLOW_WORKERS=1
export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false
export MLFLOW_SERVER_ALLOWED_HOSTS='localhost:*,127.0.0.1:*'
export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS='http://localhost:*,http://127.0.0.1:*'
if [[ -n ${CODESPACE_NAME:-} && -n ${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-} ]]; then
    _ELECTION_FORWARDED_HOST="${CODESPACE_NAME}-5000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}"
    export MLFLOW_SERVER_ALLOWED_HOSTS="$MLFLOW_SERVER_ALLOWED_HOSTS,$_ELECTION_FORWARDED_HOST"
    export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="$MLFLOW_SERVER_CORS_ALLOWED_ORIGINS,https://$_ELECTION_FORWARDED_HOST"
fi
unset _ELECTION_ENV_ROOT _ELECTION_ENV_STATE _ELECTION_FORWARDED_HOST
