-- scripts/init_governance_db.sql
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    drift_type VARCHAR(64),
    severity VARCHAR(16),
    details JSONB,
    retraining_triggered BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(64),
    user_id VARCHAR(64),
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
