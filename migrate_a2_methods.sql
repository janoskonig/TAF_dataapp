-- Migration: store both A2 ridge-height methods side by side.
--   A2_methodB: b) ortogonális (gerinc + bukkális + linguális, keresztmetszeti)
--   A2_methodC: c) ribbon (gerinc-ribbon + egyesített áthajlás-ribbon felület)
-- A2_mag_mm / A2_modszer / A2_profil az AKTÍV (utoljára számolt) módszert tükrözik.
-- Az A2_modszer 'B' vagy 'C' lehet.
-- Megjegyzés: az 'a) vertikális' módszert elhagytuk. Ha korábban létrejött az
-- A2_methodA oszlop, az használaton kívül marad (nem írjuk); igény esetén:
--   ALTER TABLE patients DROP COLUMN IF EXISTS "A2_methodA";

ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_methodB" DECIMAL(10,2) NULL;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS "A2_methodC" DECIMAL(10,2) NULL;
