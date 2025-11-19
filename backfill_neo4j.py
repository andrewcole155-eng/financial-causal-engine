import json
import logging
import os
from database_manager import DatabaseManager
from typing import Dict, Any
from datetime import datetime, timedelta

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_config() -> Dict[str, Any]:
    """
    Prioritizes Environment Variables (for GitHub Actions).
    Falls back to config.json (for Local testing).
    """
    # 1. Try Environment Variables first (Best for GitHub Actions)
    if os.environ.get("NEO4J_URI"):
        logger.info("🔧 Configuration loaded from Environment Variables.")
        return {
            "neo4j_uri": os.environ.get("NEO4J_URI"),
            "neo4j_user": os.environ.get("NEO4J_USER"),
            "neo4j_password": os.environ.get("NEO4J_PASSWORD"),
            # We assume the SQLite DB is in the root folder on the runner
            "database_path": "financial_data.db" 
        }

    # 2. Fallback to config.json
    try:
        with open('config.json', 'r') as f:
            logger.info("📂 Configuration loaded from config.json.")
            return json.load(f)
    except FileNotFoundError:
        logger.error("❌ No config found (Env Vars missing and config.json missing).")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding config.json.")
        return {}

def run_backfill():
    """
    Main script function.
    - Connects to databases using DatabaseManager.
    - Reads all events from SQLite.
    - Writes each event as a node in Neo4j.
    - **FIXES bad/null timestamps.**
    """
    logger.info("🚀 Starting Neo4j backfill script...")
    
    # 1. Load config and initialize DatabaseManager
    config = get_db_config()
    
    if not config or not config.get("neo4j_uri"):
        logger.critical("❌ Missing NEO4J_URI in config. Exiting.")
        return

    logger.warning("="*50)
    logger.warning("IMPORTANT: Connecting to Neo4j database at:")
    logger.warning(f" {config.get('neo4j_uri')}")
    logger.warning("="*50)

    db_manager = DatabaseManager(config)

    # 2. Check if both databases are connected
    if not db_manager.sqlite_conn:
        logger.critical("❌ Failed to connect to local SQLite database. Cannot get data. Exiting.")
        return
    if not db_manager.is_connected():
        logger.critical("❌ Failed to connect to Neo4j. Cannot write data. Exiting.")
        return

    # 3. NEW STEP: Clear all old/bad events from Neo4j first
    try:
        logger.info("Cleaning up old events in Neo4j...")
        db_manager.clear_neo4j_events()
    except Exception as e:
        logger.error(f"Could not clear old events from Neo4j: {e}. Stopping backfill.")
        return

    # 4. Get all events from SQLite
    # This grabs the fresh data that worker.py just put there
    all_events = db_manager.get_all_events_from_sqlite()
    if not all_events:
        logger.warning("No events found in SQLite. Nothing to backfill.")
        db_manager.close()
        return

    logger.info(f"Found {len(all_events)} events to backfill. Starting process...")

    # 5. Loop and insert into Neo4j
    count = 0
    now = datetime.now()
    total_events = len(all_events)
    
    for i, event in enumerate(all_events):
        try:
            # ### TIMESTAMP FIX ###
            # Check if the timestamp from SQLite is None or invalid
            original_timestamp = event.get('timestamp')
            
            if not original_timestamp:
                # Create a NEW, FAKE, STAGGERED timestamp
                minutes_ago = (total_events - i) * 10  # Stagger by 10 mins
                new_timestamp_dt = now - timedelta(minutes=minutes_ago)
                final_timestamp = new_timestamp_dt.isoformat()
            else:
                # The timestamp is good, use it
                final_timestamp = original_timestamp
            
            # Use the private function we created earlier
            db_manager._add_event_node_to_graph(
                ticker=event['ticker'],
                headline=event['headline'],
                score=event['score'],
                link=event['link'],
                timestamp=final_timestamp 
            )
            count += 1
            if count % 100 == 0:
                logger.info(f" -> Progress: {count}/{total_events} events processed.")
        
        except Exception as e:
            logger.error(f"Failed to add event for {event.get('ticker')}: {e}")
            # This can happen if the (c:Company) node doesn't exist
            logger.warning(f" -> Make sure a Company node with ticker '{event.get('ticker')}' exists first.")

    # 6. Clean up and report
    logger.info(f"✅ Backfill complete! {count} events processed and timestamps fixed.")
    db_manager.close()

if __name__ == "__main__":
    run_backfill()