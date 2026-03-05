-- Migration script to add F9 column
-- F9: A garatreflex erőssége (The strength of the gag reflex)
-- This should be collected in questionnaire3 (follow-up visit)

-- Add the F9 column (INTEGER to store values 1, 2, or 3)
ALTER TABLE patients ADD COLUMN F9 INT NULL;

-- Note: F9 values:
-- 1 = a páciens elmondása szerint sincs
-- 2 = a páciens elmondása szerint van, de a kezelést nem befolyásolta
-- 3 = a páciens elmondása szerint van és a kezelést jelentősen befolyásolta


