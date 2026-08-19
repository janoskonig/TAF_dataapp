BEGIN;

ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS paciens_telefonszam TEXT;

CREATE SEQUENCE IF NOT EXISTS patients_id_seq;
ALTER SEQUENCE patients_id_seq OWNED BY patients.id;
SELECT setval(
    'patients_id_seq',
    COALESCE((SELECT MAX(id) FROM patients), 0) + 1,
    false
);
ALTER TABLE patients
    ALTER COLUMN id SET DEFAULT nextval('patients_id_seq');

CREATE TABLE IF NOT EXISTS baseline_visits (
    id BIGSERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL UNIQUE REFERENCES patients(id),
    status TEXT NOT NULL DEFAULT 'registered'
        CHECK (status IN ('registered', 'in_progress', 'completed')),
    consent_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    data_collector TEXT,
    questionnaire_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    anatomy_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    init_mai_huedegree DOUBLE PRECISION,
    init_image_path TEXT,
    mai_note TEXT,
    questionnaire_completed_at TIMESTAMP,
    anatomy_completed_at TIMESTAMP,
    model_completed_at TIMESTAMP,
    mai_completed_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS baseline_visits_status_idx
    ON baseline_visits(status);

COMMIT;
