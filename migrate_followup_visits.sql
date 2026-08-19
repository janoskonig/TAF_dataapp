-- PREDICT longitudinális utánkövetés
--
-- Az utánkövetési adatok külön táblába kerülnek. A patients tábla meglévő
-- sorait ez a migráció és az új felület sem módosítja.

CREATE TABLE IF NOT EXISTS followup_visits (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    visit_round SMALLINT NOT NULL DEFAULT 1 CHECK (visit_round > 0),
    visit_status TEXT NOT NULL DEFAULT 'not_contacted' CHECK (
        visit_status IN (
            'not_contacted', 'contacted', 'scheduled', 'arrived',
            'completed', 'declined', 'no_show', 'unreachable'
        )
    ),
    contact_attempted_at TIMESTAMP NULL,
    appointment_at TIMESTAMP NULL,
    patient_display_name TEXT NULL,
    patient_phone TEXT NULL,
    contact_note TEXT NULL,
    nonattendance_reason TEXT NULL,

    consent_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    data_collector TEXT NULL,
    months_since_delivery NUMERIC(6,1) NULL CHECK (months_since_delivery >= 0),
    interim_adjustments TEXT NULL,
    interim_events TEXT NULL,

    responsiveness_today_situation_recall TEXT NULL,
    responsiveness_change TEXT NULL,
    chewing_today_situation_recall TEXT NULL,
    chewing_change TEXT NULL,

    ohip_1_recall SMALLINT NULL CHECK (ohip_1_recall BETWEEN 0 AND 4),
    ohip_2_recall SMALLINT NULL CHECK (ohip_2_recall BETWEEN 0 AND 4),
    ohip_3_recall SMALLINT NULL CHECK (ohip_3_recall BETWEEN 0 AND 4),
    ohip_4_recall SMALLINT NULL CHECK (ohip_4_recall BETWEEN 0 AND 4),
    ohip_5_recall SMALLINT NULL CHECK (ohip_5_recall BETWEEN 0 AND 4),

    gohai_1_recall SMALLINT NULL CHECK (gohai_1_recall BETWEEN 1 AND 5),
    gohai_2_recall SMALLINT NULL CHECK (gohai_2_recall BETWEEN 1 AND 5),
    gohai_3_recall SMALLINT NULL CHECK (gohai_3_recall BETWEEN 1 AND 5),
    gohai_4_recall SMALLINT NULL CHECK (gohai_4_recall BETWEEN 1 AND 5),
    gohai_5_recall SMALLINT NULL CHECK (gohai_5_recall BETWEEN 1 AND 5),
    gohai_6_recall SMALLINT NULL CHECK (gohai_6_recall BETWEEN 1 AND 5),
    gohai_7_recall SMALLINT NULL CHECK (gohai_7_recall BETWEEN 1 AND 5),
    gohai_8_recall SMALLINT NULL CHECK (gohai_8_recall BETWEEN 1 AND 5),
    gohai_9_recall SMALLINT NULL CHECK (gohai_9_recall BETWEEN 1 AND 5),
    gohai_10_recall SMALLINT NULL CHECK (gohai_10_recall BETWEEN 1 AND 5),
    gohai_11_recall SMALLINT NULL CHECK (gohai_11_recall BETWEEN 1 AND 5),
    gohai_12_recall SMALLINT NULL CHECK (gohai_12_recall BETWEEN 1 AND 5),
    questionnaire_completed_at TIMESTAMP NULL,

    f9 SMALLINT NULL CHECK (f9 BETWEEN 1 AND 3),
    final_mai DOUBLE PRECISION NULL,
    final_mai_huedegree DOUBLE PRECISION NULL,
    final_image_path TEXT NULL,
    mai_completed_at TIMESTAMP NULL,
    mai_note TEXT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,

    UNIQUE (patient_id, visit_round)
);

CREATE INDEX IF NOT EXISTS followup_visits_appointment_idx
    ON followup_visits (appointment_at);

CREATE INDEX IF NOT EXISTS followup_visits_status_idx
    ON followup_visits (visit_status);

-- A már létező utánkövetési táblák biztonságos bővítése.
ALTER TABLE followup_visits
    ADD COLUMN IF NOT EXISTS patient_display_name TEXT NULL;

ALTER TABLE followup_visits
    ADD COLUMN IF NOT EXISTS patient_phone TEXT NULL;

CREATE TABLE IF NOT EXISTS followup_contact_attempts (
    id BIGSERIAL PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    attempted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    outcome TEXT NOT NULL CHECK (
        outcome IN (
            'contacted', 'scheduled', 'declined', 'no_show',
            'unreachable', 'arrived'
        )
    ),
    note TEXT NULL
);

CREATE INDEX IF NOT EXISTS followup_contact_attempts_patient_idx
    ON followup_contact_attempts (patient_id, attempted_at DESC);
