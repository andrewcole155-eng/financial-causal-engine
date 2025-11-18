import json
import logging
from database_manager import DatabaseManager
from typing import Dict, Any
from datetime import datetime, timedelta

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Loads config.json from the same directory."""
    try:
        # ### FIX: Ensure config.json is in the project root
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
    - **FIXES bad/null timestamps.**
    """
    logger.info("🚀 Starting Neo4j backfill script...")
    
    # 1. Load config and initialize DatabaseManager
    # ### FIX: Load config from the correct "flat" format (e.g., neo4j_uri)
    config = load_config()
    if not config:
        return

    # ### FIX: Make sure your config.json points to your CLOUD Neo4j DB
    logger.warning("="*50)
    logger.warning("IMPORTANT: Make sure your config.json is pointing to your")
    logger.warning(f" CLOUD Neo4j database: {config.get('neo4j_uri')}")
    logger.warning("="*50)

    db_manager = DatabaseManager(config)

    # 2. Check if both databases are connected
    if not db_manager.sqlite_conn:
        logger.critical("❌ Failed to connect to local SQLite database. Cannot get data. Exiting.")
        return
    if not db_manager.is_connected():
        logger.critical("❌ Failed to connect to Neo4j. Cannot write data. Exiting.")
        return

    # ### NEW STEP: Clear all old/bad events from Neo4j first
    try:
        db_manager.clear_neo4j_events()
    except Exception as e:
        logger.error(f"Could not clear old events from Neo4j: {e}. Stopping backfill.")
        return

    # 3. Get all events from SQLite
    # ### FIX: Call the new function to specifically get data from SQLite
    all_events = db_manager.get_all_events_from_sqlite()
    if not all_events:
        logger.warning("No events found in SQLite. Nothing to backfill.")
        db_manager.close()
        return

    logger.info(f"Found {len(all_events)} events to backfill. Starting process...")

    # 4. Loop and insert into Neo4j
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
                # This makes the oldest event (i=0) appear far in the past
                # and the newest event (i=total_events) appear just before "now"
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
                timestamp=final_timestamp # Pass the FIXED timestamp
            )
            count += 1
            if count % 100 == 0:
                logger.info(f" -> Progress: {count}/{total_events} events processed.")
        
        except Exception as e:
            logger.error(f"Failed to add event for {event.get('ticker')}: {e}")
            # This can happen if the (c:Company) node doesn't exist
            logger.warning(f" -> Make sure a Company node with ticker '{event.get('ticker')}' exists first.")

    # 5. Clean up and report
    logger.info(f"✅ Backfill complete! {count} events processed and timestamps fixed.")
    db_manager.close()

if __name__ == "__main__":
    run_backfill()