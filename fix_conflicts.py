import json
import logging
from database_manager import DatabaseManager

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Load Config
try:
    with open("config.json", 'r') as f:
        config = json.load(f)
except:
    print("❌ Config not found.")
    exit()

db = DatabaseManager(config)

print("\n🧹 STARTING CONFLICT RESOLUTION...")
print("Goal: If two companies have an AI edge AND a Verified edge, delete the Verified one.\n")

# 1. Count Conflicts
count_query = """
MATCH (a)-[ai_rel]->(b)
WHERE ai_rel.verification_status = 'AI_PROPOSED'
MATCH (a)-[dumb_rel]->(b)
WHERE dumb_rel.verification_status IS NULL OR dumb_rel.verification_status <> 'AI_PROPOSED'
RETURN count(dumb_rel) as conflicts
"""
result = db.execute_read(count_query)
conflicts = result[0]['conflicts']

if conflicts == 0:
    print("✅ No conflicts found! The database is clean.")
    print("If you still don't see dashed lines, the issue is purely in app.py visualization.")
else:
    print(f"⚠️ Found {conflicts} conflicting edges where Verified data is blocking AI data.")
    
    # 2. Delete the blocking edges
    delete_query = """
    MATCH (a)-[ai_rel]->(b)
    WHERE ai_rel.verification_status = 'AI_PROPOSED'
    MATCH (a)-[dumb_rel]->(b)
    WHERE dumb_rel.verification_status IS NULL OR dumb_rel.verification_status <> 'AI_PROPOSED'
    
    // Only delete if they are distinct relationships
    WHERE elementId(ai_rel) <> elementId(dumb_rel)
    
    DELETE dumb_rel
    RETURN count(*) as deleted
    """
    
    write_result = db.execute_write(delete_query)
    print(f"🔥 DELETED {write_result['deleted']} blocking edges.")
    print("   The AI edges are now the ONLY connection between these nodes.")

db.close()