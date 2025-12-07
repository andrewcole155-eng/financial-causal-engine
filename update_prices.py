import os
import logging
import json
import yfinance as yf
from neo4j import GraphDatabase # <--- Direct Import
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HELPER: CONFIG LOADER ---
def load_local_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load config.json: {e}")
    return {}

def get_db_config():
    """Robustly determines credentials."""
    neo_uri = os.environ.get('NEO4J_URI')
    neo_user = os.environ.get('NEO4J_USER')
    neo_pass = os.environ.get('NEO4J_PASSWORD')

    if neo_uri and neo_user and neo_pass:
        return {"uri": neo_uri, "user": neo_user, "password": neo_pass}

    loaded = load_local_config()
    
    # Handle various nesting structures
    if "neo4j" in loaded:
        conf = loaded["neo4j"]
        return {"uri": conf.get("uri"), "user": conf.get("user"), "password": conf.get("password")}
    
    if "neo4j_uri" in loaded:
        return {
            "uri": loaded.get("neo4j_uri"),
            "user": loaded.get("neo4j_user"),
            "password": loaded.get("neo4j_password")
        }
        
    return {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

def update_live_prices():
    logger.info("🚀 Starting Live Price Update for Neo4j...")
    
    # 1. Load Config
    creds = get_db_config()
    
    # 2. Establish DIRECT Connection (Bypassing DatabaseManager wrapper issues)
    try:
        driver = GraphDatabase.driver(creds["uri"], auth=(creds["user"], creds["password"]))
        driver.verify_connectivity()
        logger.info("✅ Direct Neo4j Connection Established.")
    except Exception as e:
        logger.error(f"❌ Could not connect to Neo4j: {e}")
        return

    # 3. Get All Tickers
    query = "MATCH (c:Company) RETURN c.ticker as ticker"
    try:
        with driver.session() as session:
            results = session.run(query).data()
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        driver.close()
        return

    if not results:
        logger.warning("⚠️ No companies found in the database.")
        driver.close()
        return
        
    tickers = [row['ticker'] for row in results]
    logger.info(f"📋 Found {len(tickers)} assets in Knowledge Graph.")
    
    # 4. Batch Download Prices
    yahoo_map = {t: t.replace('.', '-') for t in tickers}
    yahoo_tickers = list(yahoo_map.values())
    
    logger.info("⬇️ Downloading latest prices...")
    try:
        data = yf.download(yahoo_tickers, period="1d", group_by='ticker', progress=True, threads=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        driver.close()
        return

    # 5. Prepare Updates
    updates = []
    is_multi = len(yahoo_tickers) > 1

    for db_ticker, yahoo_ticker in yahoo_map.items():
        try:
            if is_multi:
                if yahoo_ticker in data.columns.levels[0]:
                    df = data[yahoo_ticker]
                else:
                    continue
            else:
                df = data
                
            if not df.empty and 'Close' in df.columns:
                valid_closes = df['Close'].dropna()
                if not valid_closes.empty:
                    last_price = float(valid_closes.iloc[-1])
                    if last_price > 0:
                        updates.append({"ticker": db_ticker, "price": last_price})
        except Exception:
            continue

    # 6. Write to Neo4j (Direct Session)
    if updates:
        logger.info(f"💾 Updating {len(updates)} prices in Neo4j...")
        
        write_query = """
        UNWIND $batch as row
        MATCH (c:Company {ticker: row.ticker})
        SET c.last_close = row.price,
            c.last_updated = datetime()
        """
        
        try:
            with driver.session() as session:
                session.run(write_query, batch=updates)
            logger.info("✅ Success! Prices updated.")
        except Exception as e:
            logger.error(f"Database write error: {e}")
    else:
        logger.warning("No valid prices found to update.")

    driver.close()

if __name__ == "__main__":
    update_live_prices()