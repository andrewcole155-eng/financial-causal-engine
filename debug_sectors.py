from database_manager import DatabaseManager
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Debug")

def load_config():
    try:
        with open("config.json", 'r') as f: return json.load(f)
    except: return {}

def check_sectors():
    config = load_config()
    if not config: return
    
    db = DatabaseManager(config)
    
    print("\n📊 --- SECTOR CENSUS ---")
    
    # 1. Count Nodes by Sector
    query = """
    MATCH (n:Company)
    RETURN n.sector as Sector, count(n) as Count
    ORDER BY Count DESC
    """
    results = db.execute_read(query)
    
    for r in results:
        sec = r['Sector'] if r['Sector'] else "⛔ NULL/MISSING"
        print(f"{sec:<30} : {r['Count']}")

    print("\n🔍 --- SAMPLE OF 'UNKNOWN' NODES ---")
    # 2. See who the 'Unknowns' are
    query_unknown = """
    MATCH (n:Company)
    WHERE n.sector = 'Unknown' OR n.sector IS NULL
    RETURN n.ticker as Ticker, n.name as Name
    LIMIT 10
    """
    unknowns = db.execute_read(query_unknown)
    for u in unknowns:
        print(f"{u['Ticker']} ({u['Name']})")

    db.close()

if __name__ == "__main__":
    check_sectors()