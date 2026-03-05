#!/usr/bin/env python3
"""
Creates the student_examinations table in PostgreSQL for the spinoff study
comparing experienced dentist vs dental student clinical anatomy assessments.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2

load_dotenv(dotenv_path=".env")

database_url = os.getenv("DATABASE_URL")
if not database_url:
    sys.exit("DATABASE_URL not found in .env")


def main():
    print("=== Creating student_examinations table ===\n")

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'student_examinations'
        )
    """)
    if cur.fetchone()[0]:
        print("Table 'student_examinations' already exists. Skipping creation.")
        cur.close()
        conn.close()
        return

    create_sql = """
    CREATE TABLE student_examinations (
        id SERIAL PRIMARY KEY,
        TAJ VARCHAR(11) NOT NULL,
        student_name VARCHAR(50) NOT NULL,
        denture_type VARCHAR(10) NOT NULL,
        F5 INTEGER,
        F7 INTEGER,
        F8 INTEGER,
        A1_Kaan INTEGER,
        A3_jobb INTEGER,
        A3_bal INTEGER,
        A4_jobb INTEGER,
        A4_bal INTEGER,
        A5_jobb INTEGER,
        A5_bal INTEGER,
        A6_jobb INTEGER,
        A6_bal INTEGER,
        A7_jobb INTEGER,
        A7_bal INTEGER,
        A8_jobb INTEGER,
        A8_bal INTEGER,
        A9_jobb INTEGER,
        A9_bal INTEGER,
        A11 INTEGER,
        A12 INTEGER,
        A13 INTEGER,
        A14 INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    )
    """

    print("Creating table...")
    cur.execute(create_sql)
    conn.commit()

    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'student_examinations'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    print(f"\nCreated table with {len(columns)} columns:")
    for name, dtype in columns:
        print(f"  {name:20s} {dtype}")

    cur.close()
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
