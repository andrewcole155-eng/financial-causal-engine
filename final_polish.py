import pandas as pd
import requests
import json
import os
import datetime  # <--- ADD THIS IMPORT
from io import StringIO
from neo4j import GraphDatabase

def load_config():
    config_path = 'config.json'
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def fix_sector_and_export():
    config = load_config()
    neo4j_uri = config.get("neo4j_uri", "neo4j+s://da0b2bf9.databases.neo4j.io")
    neo4j_user = config.get("neo4j_user", "neo4j")
    neo4j_password = config.get("neo4j_password")

    # 1. Fetch Wikipedia Data
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    print(f"📡 Fetching fresh data from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        df = None
        for t in tables:
            if "Symbol" in t.columns and "GICS Sector" in t.columns:
                df = t
                break
        
        if df is None: return

        master_map = {}
        for _, row in df.iterrows():
            ticker = row['Symbol']
            master_map[ticker] = {"name": row['Security'], "sector": row['GICS Sector']}
            master_map[ticker.replace('.', '-')] = {"name": row['Security'], "sector": row['GICS Sector']}

        manual_additions = {
            "IONQ": {"name": "IonQ, Inc.", "sector": "Technology"},
            "SNAP": {"name": "Snap Inc.", "sector": "Communication Services"},
            "SIRI": {"name": "Sirius XM Holdings", "sector": "Communication Services"},
            "COR":  {"name": "Cencora", "sector": "Health Care"},
            "CPAY": {"name": "Corpay", "sector": "Financials"},
            "DOC":  {"name": "Healthpeak Properties", "sector": "Real Estate"},
            "KKR":  {"name": "KKR & Co. Inc.", "sector": "Financials"},
            "GE":   {"name": "GE Aerospace", "sector": "Industrials"}
        }
        master_map.update(manual_additions)

        # 2. Export from Neo4j
        print(f"🔌 Connecting to Neo4j...")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        query = """
        MATCH (c:Company)
        WHERE c.gnn_risk_score IS NOT NULL
        RETURN c.ticker AS Ticker, c.gnn_risk_score AS Risk_Score, c.gnn_last_updated AS Last_Updated
        ORDER BY c.gnn_risk_score DESC
        """
        
        export_data = []
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # <--- Capture time now
        
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                ticker = record['Ticker']
                if ticker in master_map:
                    info = master_map[ticker]
                    
                    # Fix None timestamp
                    last_updated = record['Last_Updated']
                    if not last_updated:
                        last_updated = current_time

                    export_data.append({
                        "Ticker": ticker,
                        "Name": info['name'],
                        "Sector": info['sector'],
                        "Risk_Score": record['Risk_Score'],
                        "Last_Updated": last_updated # <--- Use the fixed time
                    })
        
        driver.close()
        
        if export_data:
            df_out = pd.DataFrame(export_data)
            df_out.to_csv("live_risk_scores.csv", index=False)
            print(f"🎉 Success! Saved {len(df_out)} records to live_risk_scores.csv")
            print(df_out.head())

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_sector_and_export()