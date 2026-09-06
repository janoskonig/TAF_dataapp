-- PREDICT szakértői prior-felmérés
--
-- A szakértői válaszok külön táblába kerülnek; a patients és a vizit-táblák
-- sorait ez a migráció nem érinti. Idempotens: többször is futtatható.

BEGIN;

CREATE TABLE IF NOT EXISTS expert_prior_responses (
    id BIGSERIAL PRIMARY KEY,
    expert_code TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'submitted')),
    consent_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    background JSONB NOT NULL DEFAULT '{}'::jsonb,
    calibration JSONB NOT NULL DEFAULT '{}'::jsonb,
    items JSONB NOT NULL DEFAULT '{}'::jsonb,
    closing JSONB NOT NULL DEFAULT '{}'::jsonb,
    form_version TEXT NOT NULL DEFAULT 'v1.0',
    submitted_at TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS expert_prior_responses_status_idx
    ON expert_prior_responses(status);

-- 2026-09-06: a kitöltő szakértő neve és munkahelye (vizsgálatvezetői döntés:
-- rögzíteni kell, kit kérdezünk). Az exportban a kód marad az azonosító.
ALTER TABLE expert_prior_responses
    ADD COLUMN IF NOT EXISTS expert_name TEXT,
    ADD COLUMN IF NOT EXISTS expert_affiliation TEXT;

-- 2026-09-06: személyes meghívó-linkek (a vizsgálatvezető készíti elő a
-- kitöltést; a link kód nélkül, csak az adott kitöltésbe enged be).
ALTER TABLE expert_prior_responses
    ADD COLUMN IF NOT EXISTS invited_at TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS opened_at TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS invite_note TEXT NULL;

COMMIT;
