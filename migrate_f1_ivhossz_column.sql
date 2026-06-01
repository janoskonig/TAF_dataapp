-- Migration: felső gerincél 3D ívhossza (mm).
-- Az F2 alámenősség (mm³) méretfüggetlen standardizálásához: F2 / L³.
-- A Blender addon az F1 méréskor számolja a gerincgörbéből (fix 2 mm-es
-- X-binelés + per-bin medián, majd 3D szakaszhosszak összege).

ALTER TABLE patients ADD COLUMN IF NOT EXISTS "F1_ivhossz_mm" DECIMAL(10,2) NULL;
