import sqlite3

conn = sqlite3.connect("data/app.db")
for table in ("users", "history"):
    print(f"\n=== {table} ===")
    cur = conn.execute(f"SELECT * FROM {table}")
    print([c[0] for c in cur.description])
    for row in cur.fetchall():
        print(row)
conn.close()