import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "user_study.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Example: see the first 20 rows from evaluation_ratings
cur.execute("SELECT * FROM evaluation_ratings ORDER BY timestamp DESC LIMIT 20")
rows = cur.fetchall()

print(f"Found {len(rows)} rows")
for row in rows:
    print(dict(row))

conn.close()