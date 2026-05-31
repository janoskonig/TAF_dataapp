-- Migration: add numeric A2 ridge height column
-- A2_mag_mm: alsó gerinc magassága mm-ben (a Blender addon méri)
-- Distinct from A2_gerincelvonal / A2_bukkalisathajlas / A2_lingualisathajlas
-- which store FTP paths to STL files.

ALTER TABLE patients ADD COLUMN "A2_mag_mm" DECIMAL(10,2) NULL;
ALTER TABLE patients ADD COLUMN "A2_modszer" VARCHAR(1)   NULL;  -- 'A' or 'B'
