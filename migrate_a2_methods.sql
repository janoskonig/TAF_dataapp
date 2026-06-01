-- Migration: store all three A2 ridge-height methods side by side.
--   A2_methodA: a) vertikális (gerinc + 1 áthajlás)
--   A2_methodB: b) ortogonális (gerinc + bukkális + linguális, keresztmetszeti)
--   A2_methodC: c) ribbon (gerinc-ribbon + egyesített áthajlás-ribbon felület)
-- A2_mag_mm / A2_modszer / A2_profil továbbra is az AKTÍV (utoljára számolt) módszert tükrözik.
-- Az A2_modszer mostantól 'A', 'B' vagy 'C' lehet (a VARCHAR(1) ezt eleve engedi).

ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_methodA" DECIMAL(10,2) NULL;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_methodB" DECIMAL(10,2) NULL;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_methodC" DECIMAL(10,2) NULL;
