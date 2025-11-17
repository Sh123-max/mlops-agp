#!/usr/bin/env bash
set -euo pipefail

# scripts/init_db.sh
# Usage:
#   ./scripts/init_db.sh
# or override env vars:
#   PG_CONTAINER_NAME=mlops-agp-postgres-1 PGUSER=postgres PGDB=postgres PGPASSWORD=secret ./scripts/init_db.sh

# --- configurable ---
PG_CONTAINER_NAME="${PG_CONTAINER_NAME:-mlops-agp-postgres-1}"  # docker-compose postgres container name (adjust if different)
SQL_FILE_LOCAL="./scripts/init_governance_db.sql"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-mlflow}"
PGDB="${PGDB:-mlflow_db}"
export PGPASSWORD="${PGPASSWORD:-mlflow123}"
# -----------------------

if [ ! -f "${SQL_FILE_LOCAL}" ]; then
  echo "[ERROR] SQL file not found: ${SQL_FILE_LOCAL}"
  exit 2
fi

echo "[init_db] SQL file: ${SQL_FILE_LOCAL}"
echo "[init_db] Target PG container (env PG_CONTAINER_NAME) = ${PG_CONTAINER_NAME}"
echo "[init_db] Trying to detect Docker container named ${PG_CONTAINER_NAME}..."

# check for a running container with that name
CONTAINER_ID="$(docker ps -q -f name=^/${PG_CONTAINER_NAME}$ || true)"

if [ -n "${CONTAINER_ID}" ]; then
  echo "[init_db] Found running container ${PG_CONTAINER_NAME} (id=${CONTAINER_ID}). Copying SQL and executing inside container..."
  docker cp "${SQL_FILE_LOCAL}" "${CONTAINER_ID}:/tmp/init_governance_db.sql"
  # run as postgres user by default; allow PGUSER override
  docker exec -i "${CONTAINER_ID}" bash -lc "psql -U ${PGUSER} -d ${PGDB} -f /tmp/init_governance_db.sql"
  echo "[init_db] SQL executed inside container ${PG_CONTAINER_NAME}."
  exit 0
fi

echo "[init_db] No running container named ${PG_CONTAINER_NAME}. Attempting to connect to Postgres on host ${PGHOST}:${PGPORT}."

# Attempt local psql invocation
if command -v psql >/dev/null 2>&1; then
  echo "[init_db] Running psql locally: psql -h ${PGHOST} -p ${PGPORT} -U ${PGUSER} -d ${PGDB} -f ${SQL_FILE_LOCAL}"
  psql -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${PGDB}" -f "${SQL_FILE_LOCAL}"
  echo "[init_db] SQL executed on host Postgres."
  exit 0
else
  echo "[ERROR] psql not found on host and container ${PG_CONTAINER_NAME} is not running."
  echo "Install psql or start your postgres container, or set PG_CONTAINER_NAME to the running container name."
  exit 3
fi

