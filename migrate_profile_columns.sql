-- Migration: dedicated profile point storage for F1 and A2
-- The Blender addon sends the raw ridge profile (per-point arrays) as JSON.
-- We store that JSON text in its own columns, separate from the legacy
-- *_gerincelvonal path columns (which the web form still uses for STL paths).
--
-- Value is a JSON array of {x, z, zref, d} points (A2 method-b points also
-- carry a "side": "buccal"/"lingual" tag).

ALTER TABLE patients ADD COLUMN IF NOT EXISTS "F1_profil" TEXT NULL;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_profil" TEXT NULL;

-- Reading in pandas:
--   df['F1_profil'].dropna().map(json.loads)
-- or in SQL:
--   SELECT "F1_profil"::jsonb FROM patients WHERE "F1_profil" IS NOT NULL;
