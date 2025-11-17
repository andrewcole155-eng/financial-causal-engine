import json
import logging
import networkx as nx
from database_manager import DatabaseManager
import os
import sys

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_config_from_secrets():
    """
    Reads Neo4j credentials from GitHub environment secrets
    and writes them to a local config.json file.
    """
    logger.info("Loading secrets from environment...")
    
    NEO4J_URI = os.environ.get('NEO4J_URI')
    NEO4J_USER = os.environ.get('NEO4J_USER')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Missing one or more Neo4j credentials in environment variables.")
        return False

    # This is the config structure your DatabaseManager is likely expecting
    config_data = {
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": NEO4J_PASSWORD
        }
    }

    try:
        # Note: We are writing to 'config.json' in the root,
        # which is where your DatabaseManager appears to be looking.
        with open("config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        logger.info("Successfully created temporary config.json")
        return True
    except Exception as e:
        logger.error(f"Failed to write config.json: {e}")
        return False

def main():
    logger.info("Starting pre-computation job...")
    
    # 1. Create the config.json file
    if not create_config_from_secrets():
        logger.critical("Could not create config from secrets. Exiting.")
        sys.exit(1) # Exit with an error

    # 2. Initialize DatabaseManager
    # We pass NO arguments, so it will find and read config.json
    logger.info("Initializing DatabaseManager (will read from config.json)...")
    db_manager = DatabaseManager() 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. Check credentials in GitHub secrets.")
        sys.exit(1) # Exit with an error

    logger.info("Connected to Neo4j. Fetching full graph...")
    
    # 3. This is the slow part (can take a long time)
    graph = db_manager.get_graph_from_db(weight_threshold=0.1)
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty. Saving an empty graph file.")
    else:
        logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")

    # 4. Save the computed graph to a file
    output_file = "financial_graph.gml"
    nx.write_gml(graph, output_file)
    
    logger.info(f"Successfully saved graph to {output_file}")
    
    # 5. Clean up the temporary config file (optional, but good practice)
    try:
        os.remove("config.json")
        logger.info("Cleaned up temporary config.json")
    except Exception as e:
        logger.warning(f"Could not remove config.json: {e}")

if __name__ == "__main__":
    main()