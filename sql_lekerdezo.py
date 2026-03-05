"""
Ez a program egy SQL adatbázis lekérdező program... akarna lenni.
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

DATABASE_URL = os.getenv("DATABASE_URL")

# Kapcsolódás az adatbázishoz
connection = psycopg2.connect(DATABASE_URL)

# Lekérdezés
cursor = connection.cursor()
cursor.execute("SELECT * FROM patients")
result = cursor.fetchall()
print(result[-1])