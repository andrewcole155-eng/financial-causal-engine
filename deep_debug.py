import json
from neo4j import GraphDatabase

# 1. Load Config directly
try:
    with open("config.json", 'r') as f:
        config = json.load(f)
    uri = config.get("neo4j_uri") or config["neo4j"]["uri"]
    user = config.get("neo4j_user") or config["neo4j"]["user"]
    password = config.get("neo4j_password") or config["neo4j"]["password"]
except Exception as e:
    print(f"❌ Config Error: {e}")
    exit()

print(f"🔌 Connecting to: {uri} as {user}")

# 2. Connect Raw Driver
driver = GraphDatabase.driver(uri, auth=(user, password))

query = """
MATCH (a:Company {ticker: 'T'})-[r]-(b:Company {ticker: 'VZ'})
RETURN type(r) as type, properties(r) as props, startNode(r).ticker as source, endNode(r).ticker as target
"""

with driver.session() as session:
    results = session.run(query).data()
    
    print(f"\nFound {len(results)} raw connections between T and VZ:\n")
    
    for i, row in enumerate(results):
        print(f"--- Connection {i+1} ---")
        print(f"Type: {row['type']}")
        print(f"Direction: {row['source']} -> {row['target']}")
        print(f"RAW PROPERTIES: {row['props']}")
        
        status = row['props'].get('verification_status')
        print(f"👉 Verification Status: {status}")
        
        if status == 'AI_PROPOSED':
            print("   ✅ This is the AI Edge.")
        else:
            print("   ❌ This is a Verified/Ghost Edge.")
            
driver.close()