#!/usr/bin/env bash
set -e
# Usage: ./scripts/init_db.sh
PGHOST=${PGHOST:-localhost}
PGPORT=${PGPORT:-5432}
PGUSER=${PGUSER:-mlflow}
PGDB=${PGDB:-mlflow_db}
export PGPASSWORD=${PGPASSWORD:-mlflow123}

if [ -n "$(docker ps -q -f name=postgres)" ]; then
  docker cp scripts/init_governance_db.sql $(docker ps -q -f name=postgres):/init_governance_db.sql
  docker exec -i $(docker ps -q -f name=postgres) psql -U $PGUSER -d $PGDB -f /init_governance_db.sql
else
  psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDB -f scripts/init_governance_db.sql
fi

echo "Governance tables initialized."
