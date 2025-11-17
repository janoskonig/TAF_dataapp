"""
Ez a program egy SQL adatbázis lekérdező program... akarna lenni.
"""

import mysql.connector
import os

host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

# Kapcsolódás az adatbázishoz
connection = mysql.connector.connect(
    host=host,
    user=user,
    port=port,
    password=password,
    database=database
)

# Lekérdezés
cursor = connection.cursor()
cursor.execute("SELECT * FROM patients")
result = cursor.fetchall()
print(result[-1])