#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${DB_HOST:=localhost}"
: "${DB_PORT:=5432}"
: "${DB_USER:=crm_app}"
: "${DB_PASSWORD:=crm_password}"
: "${DB_NAME:=crm_sandbox}"

SQL_FILE="${REPO_ROOT}/sql/02_seed_data.sql"
SQL_CONTAINER_PATH="/app/sql/02_seed_data.sql"
USE_COMPOSE="${DB_SEED_USE_COMPOSE:-auto}"

if [[ ! -f "${SQL_FILE}" ]]; then
  echo "Seed SQL file not found at ${SQL_FILE}" >&2
  exit 1
fi

try_compose_exec() {
  if ! command -v docker >/dev/null; then
    return 1
  fi
  if ! docker compose ps db >/dev/null 2>&1; then
    return 1
  fi
  echo "Applying seed data via docker compose exec..."
  # SQL file now has explicit BEGIN/COMMIT, so we don't need --single-transaction
  # but we keep ON_ERROR_STOP=on to fail fast on errors
  docker compose exec -T db psql \
    --username "${POSTGRES_USER:-$DB_USER}" \
    --dbname "${POSTGRES_DB:-$DB_NAME}" \
    --file "${SQL_CONTAINER_PATH}" \
    --echo-errors \
    --set ON_ERROR_STOP=on
}

should_use_compose() {
  [[ "${USE_COMPOSE}" == "always" ]] && return 0
  if [[ "${USE_COMPOSE}" == "never" ]]; then
    return 1
  fi
  case "${DB_HOST}" in
    localhost|127.0.0.1|db) return 0 ;;
    *) return 1 ;;
  esac
}

if should_use_compose && try_compose_exec; then
  echo "Seed data applied successfully (docker compose)."
  exit 0
fi

export PGPASSWORD="${DB_PASSWORD}"

# SQL file now has explicit BEGIN/COMMIT, so we don't need --single-transaction
# but we keep ON_ERROR_STOP=on to fail fast on errors
psql \
  --host "${DB_HOST}" \
  --port "${DB_PORT}" \
  --username "${DB_USER}" \
  --dbname "${DB_NAME}" \
  --file "${SQL_FILE}" \
  --echo-errors \
  --set ON_ERROR_STOP=on

echo "Seed data applied successfully."
