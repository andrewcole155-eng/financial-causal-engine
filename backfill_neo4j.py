# backfill_neo4j.py

import json
import logging
from database_manager import DatabaseManager
from typing import Dict, Any

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Loads config.json from the same directory."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found. Make sure it's in the same directory.")
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
    """
    logger.info("🚀 Starting Neo4j backfill script...")
    
    # 1. Load config and initialize DatabaseManager
    config = load_config()
    if not config:
        return

    db_manager = DatabaseManager(config)

    # 2. Check if both databases are connected
    if not db_manager.sqlite_conn or not db_manager.is_connected():
        logger.critical("❌ Failed to connect to one or both databases. Exiting.")
        return

    # 3. Get all events from SQLite
    all_events = db_manager.get_all_events()
    if not all_events:
        logger.warning("No events found in SQLite. Nothing to backfill.")
        db_manager.close()
        return

    logger.info(f"Found {len(all_events)} events to backfill. Starting process...")

    # 4. Loop and insert into Neo4j
    count = 0
    for event in all_events:
        try:
            # Use the private function we created earlier
            db_manager._add_event_node_to_graph(
                ticker=event['ticker'],
                headline=event['headline'],
                score=event['score'],
                link=event['link'],
                timestamp=event['timestamp'] # Pass the existing timestamp
            )
            count += 1
            if count % 100 == 0:
                logger.info(f" -> Progress: {count}/{len(all_events)} events processed.")
        
        except Exception as e:
            logger.error(f"Failed to add event for {event.get('ticker')}: {e}")

    # 5. Clean up and report
    logger.info(f"✅ Backfill complete! {count} events processed.")
    db_manager.close()

if __name__ == "__main__":
    run_backfill()