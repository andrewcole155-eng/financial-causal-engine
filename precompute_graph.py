import json
import logging
import networkx as nx
from database_manager import DatabaseManager
import os
import sys

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_config_and_get_dict():
    """
    Reads Neo4j credentials from GitHub environment secrets,
    writes them to the config.json file *in the same
    directory as database_manager.py*, AND returns them as a dictionary.
    """
    logger.info("Loading secrets from environment...")
    
    NEO4J_URI = os.environ.get('NEO4J_URI')
    NEO4J_USER = os.environ.get('NEO4J_USER')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Missing one or more Neo4j credentials in environment variables.")
        return None

    # This is the config structure your DatabaseManager is expecting
    config_data = {
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": NEO4J_PASSWORD
        }
    }

    try:
        # --- THIS IS THE FIX ---
        # Find the directory where the database_manager.py module is located
        db_module_path = os.path.dirname(os.path.abspath(DatabaseManager.__module_file__))
        config_file_path = os.path.join(db_module_path, 'config.json')
        # --- END FIX ---
        
        logger.info(f"Overwriting config file at: {config_file_path}")
        
        # Overwrite the config.json in that specific directory
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
            
        logger.info("Successfully created/overwritten config.json")
        return config_data # Return the dictionary
    except Exception as e:
        logger.error(f"Failed to find and write config.json: {e}")
        return None

def main():
    logger.info("Starting pre-computation job...")
    
    # 1. Create config.json AND get the config_dict
    cloud_config = create_config_and_get_dict()
    if not cloud_config:
        logger.critical("Could not create config from secrets. Exiting.")
        sys.exit(1) # Exit with an error

    # 2. Initialize DatabaseManager
    # We pass the config_dict just in case (to prevent the TypeError)
    logger.info("Initializing DatabaseManager...")
    db_manager = DatabaseManager(cloud_config) 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. Check credentials in GitHub secrets.")
        sys.exit(1) # Exit with an error

    logger.info("✅ Connected to Neo4j. Fetching full graph...")
    
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
    
    # 5. Clean up the config file (optional, but good practice)
    try:
        # We need the path again
        db_module_path = os.path.dirname(os.path.abspath(DatabaseManager.__module_file__))
        config_file_path = os.path.join(db_module_path, 'config.json')
        os.remove(config_file_path)
        logger.info(f"Cleaned up temporary config file at {config_file_path}")
    except Exception as e:
        logger.warning(f"Could not remove config.json: {e}")

if __name__ == "__main__":
    main()