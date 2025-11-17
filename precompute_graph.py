import json
import logging
import os
import sys

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_config_at_startup():
    """
    Reads Neo4j credentials from GitHub environment secrets
    and writes them to 'config.json' *immediately*.
    This runs BEFORE other modules are imported.
    """
    logger.info("Loading secrets from environment...")
    
    NEO4J_URI = os.environ.get('NEO4J_URI')
    NEO4J_USER = os.environ.get('NEO4J_USER')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Missing one or more Neo4j credentials in environment variables.")
        return None

    config_data = {
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": NEO4J_PASSWORD
        }
    }

    try:
        # Based on your path, config.json is in the root.
        config_file_path = "config.json"
        
        logger.info(f"Overwriting config file at: {config_file_path}")
        
        # Overwrite the config.json in that specific directory
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
            
        logger.info(f"Successfully created/overwritten {config_file_path}")
        return config_data # Return the dictionary
    except Exception as e:
        logger.error(f"Failed to write config.json: {e}")
        return None

# --- RUN THE CONFIG CREATION FIRST ---
# This code runs *immediately*
cloud_config = create_config_at_startup()
if not cloud_config:
    logger.critical("Could not create config from secrets. Exiting.")
    sys.exit(1)

# --- NOW, IMPORT THE MODULES ---
# By importing here, database_manager will read the
# config.json file we *just* created.
import networkx as nx
import database_manager 

def main():
    logger.info("Starting pre-computation job...")
    
    # 1. Initialize DatabaseManager
    # We pass the config_dict to satisfy the TypeError from earlier
    logger.info("Initializing DatabaseManager...")
    db_manager = database_manager.DatabaseManager(cloud_config) 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. Check credentials in GitHub secrets.")
        sys.exit(1) # Exit with an error

    logger.info("✅ Connected to Neo4j. Fetching full graph...")
    
    # 2. This is the slow part (can take a long time)
    graph = db_manager.get_graph_from_db(weight_threshold=0.1)
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty. Saving an empty graph file.")
    else:
        logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")

    # 3. Save the computed graph to a file
    output_file = "financial_graph.gml"
    nx.write_gml(graph, output_file)
    
    logger.info(f"Successfully saved graph to {output_file}")
    
    # 4. Clean up the config file (optional, but good practice)
    try:
        os.remove("config.json")
        logger.info("Cleaned up temporary config.json")
    except Exception as e:
        logger.warning(f"Could not remove config.json: {e}")

if __name__ == "__main__":
    main()