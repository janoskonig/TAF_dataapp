-- Migration script to refactor F1 columns
-- This script replaces F1_gerincelvonal and F1_bukkalisathajlas with a single F1 column

-- Step 1: Add the new F1 column (DECIMAL with 2 decimal places for mm values)
-- We'll add it before F2 to maintain logical ordering
ALTER TABLE patients ADD COLUMN F1 DECIMAL(10,2) NULL;

-- Step 2: Drop the old F1 columns
ALTER TABLE patients DROP COLUMN F1_gerincelvonal;
ALTER TABLE patients DROP COLUMN F1_bukkalisathajlas;

-- Optional: If you want F1 to be positioned before F2, you can run this after the above:
-- ALTER TABLE patients MODIFY COLUMN F1 DECIMAL(10,2) NULL AFTER A14;

