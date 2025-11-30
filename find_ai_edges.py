# find_ai_edges.py
import json
from database_manager import DatabaseManager

# Load Config
try:
    with open("config.json", 'r') as f:
        config = json.load(f)
except:
    print("❌ Config not found.")
    exit()

db = DatabaseManager(config)

print("\n🕵️‍♀️ SEARCHING FOR AI-INFERRED RELATIONSHIPS IN NEO4J...\n")

query = """
MATCH (a)-[r]->(b)
WHERE r.verification_status = 'AI_PROPOSED'
RETURN a.ticker, type(r), b.ticker, r.mechanism, r.weight
LIMIT 10
"""

results = db.execute_read(query)

if not results:
    print("❌ No AI edges found in the database. Run daily_batch_ingest.py again.")
else:
    print(f"✅ Found {len(results)} examples. Go look at these companies in your App:\n")
    for row in results:
        print(f"👉 GO TO: {row['a.ticker']} (Target: {row['b.ticker']})")
        print(f"   Reason: {row['r.mechanism']}")
        print("   ---------------------------------------------------")

db.close()