#!/usr/bin/env python3
"""
One-time migration script: MySQL -> PostgreSQL
Reads all data from the MySQL `patients` table and recreates it in PostgreSQL.
"""

import os
import sys
from dotenv import load_dotenv
import mysql.connector
import psycopg2
from urllib.parse import urlparse

load_dotenv(dotenv_path=".env")

# MySQL connection (source)
mysql_config = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# PostgreSQL connection (target)
database_url = os.getenv("DATABASE_URL")
if not database_url:
    sys.exit("DATABASE_URL not found in .env")

MYSQL_TO_PG_TYPE = {
    "int": "INTEGER",
    "tinyint": "SMALLINT",
    "smallint": "SMALLINT",
    "mediumint": "INTEGER",
    "bigint": "BIGINT",
    "float": "REAL",
    "double": "DOUBLE PRECISION",
    "decimal": "NUMERIC",
    "varchar": "VARCHAR",
    "char": "CHAR",
    "text": "TEXT",
    "mediumtext": "TEXT",
    "longtext": "TEXT",
    "date": "DATE",
    "datetime": "TIMESTAMP",
    "timestamp": "TIMESTAMP",
    "blob": "BYTEA",
    "tinyint(1)": "BOOLEAN",
}


def mysql_type_to_pg(mysql_type: str) -> str:
    """Convert a MySQL column type string to its PostgreSQL equivalent."""
    t = mysql_type.lower().strip()

    # tinyint(1) is typically boolean
    if t == "tinyint(1)":
        return "BOOLEAN"

    # Handle types with precision like decimal(10,2), varchar(255)
    base = t.split("(")[0]
    if base in MYSQL_TO_PG_TYPE:
        if base in ("decimal", "varchar", "char") and "(" in t:
            precision = t[t.index("("):]
            return f"{MYSQL_TO_PG_TYPE[base]}{precision}"
        return MYSQL_TO_PG_TYPE[base]

    # Fallback
    return "TEXT"


def main():
    print("=== MySQL -> PostgreSQL Migration ===\n")

    # --- Connect to MySQL ---
    print(f"Connecting to MySQL at {mysql_config['host']}:{mysql_config['port']}...")
    mysql_conn = mysql.connector.connect(**mysql_config)
    mysql_cur = mysql_conn.cursor()

    # --- Get MySQL schema ---
    mysql_cur.execute("DESCRIBE patients")
    columns_info = mysql_cur.fetchall()  # (Field, Type, Null, Key, Default, Extra)

    print(f"Found {len(columns_info)} columns in MySQL `patients` table:\n")
    col_names = []
    pg_col_defs = []
    for col in columns_info:
        name, mysql_type, nullable, key, default, extra = col
        col_names.append(name)
        pg_type = mysql_type_to_pg(mysql_type)
        parts = [f'"{name}" {pg_type}']
        if key == "PRI":
            parts.append("PRIMARY KEY")
        elif nullable == "NO":
            parts.append("NOT NULL")
        pg_col_defs.append(" ".join(parts))
        print(f"  {name:40s} {mysql_type:20s} -> {pg_type}")

    # --- Read all data from MySQL ---
    mysql_cur.execute("SELECT * FROM patients")
    rows = mysql_cur.fetchall()
    print(f"\nFetched {len(rows)} rows from MySQL.\n")

    mysql_cur.close()
    mysql_conn.close()

    # --- Connect to PostgreSQL ---
    print(f"Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(database_url)
    pg_cur = pg_conn.cursor()

    # --- Drop existing table if it exists ---
    pg_cur.execute("DROP TABLE IF EXISTS patients CASCADE")
    pg_conn.commit()

    # --- Create table in PostgreSQL ---
    create_sql = "CREATE TABLE patients (\n  " + ",\n  ".join(pg_col_defs) + "\n)"
    print("Creating PostgreSQL table...")
    print(create_sql[:200] + "...\n")
    pg_cur.execute(create_sql)
    pg_conn.commit()

    # Build a map of which columns are boolean so we can cast int -> bool
    bool_cols = set()
    for col in columns_info:
        name, mysql_type, *_ = col
        if mysql_type.lower().strip() == "tinyint(1)":
            bool_cols.add(name)

    def convert_row(row):
        converted = []
        for val, name in zip(row, col_names):
            if name in bool_cols and val is not None:
                converted.append(bool(val))
            else:
                converted.append(val)
        return tuple(converted)

    # --- Insert data ---
    if rows:
        placeholders = ", ".join(["%s"] * len(col_names))
        quoted_cols = ", ".join([f'"{c}"' for c in col_names])
        insert_sql = f"INSERT INTO patients ({quoted_cols}) VALUES ({placeholders})"

        print(f"Inserting {len(rows)} rows into PostgreSQL...")
        for i, row in enumerate(rows):
            pg_cur.execute(insert_sql, convert_row(row))
            if (i + 1) % 50 == 0:
                print(f"  ...inserted {i + 1}/{len(rows)}")
        pg_conn.commit()
        print(f"  Inserted all {len(rows)} rows.\n")
    else:
        print("No rows to insert.\n")

    # --- Verify ---
    pg_cur.execute("SELECT COUNT(*) FROM patients")
    pg_count = pg_cur.fetchone()[0]
    print(f"Verification: {pg_count} rows in PostgreSQL (expected {len(rows)})")

    if pg_count == len(rows):
        print("Migration successful!")
    else:
        print("WARNING: Row count mismatch!")

    pg_cur.close()
    pg_conn.close()


if __name__ == "__main__":
    main()
