import logging
import os
import sys

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_secrets_file_at_startup():
    """
    Reads Neo4j credentials from GitHub environment secrets
    and writes them to '.streamlit/secrets.toml' *immediately*.
    This runs BEFORE other modules are imported.
    """
    logger.info("Loading secrets from environment...")
    
    NEO4J_URI = os.environ.get('NEO4J_URI')
    NEO4J_USER = os.environ.get('NEO4J_USER')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Missing one or more Neo4j credentials in environment variables.")
        return None

    # This is the TOML format your DatabaseManager is expecting
    toml_content = f"""
[neo4j]
uri = "{NEO4J_URI}"
user = "{NEO4J_USER}"
password = "{NEO4J_PASSWORD}"
"""
    # This is the config_dict we will pass to the class constructor
    config_dict = {
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": NEO4J_PASSWORD
        }
    }

    try:
        # --- THIS IS THE FIX ---
        # The code is hard-coded to read .streamlit/secrets.toml
        config_dir = ".streamlit"
        config_file_path = os.path.join(config_dir, 'secrets.toml')
        
        # Ensure the '.streamlit' directory exists
        os.makedirs(config_dir, exist_ok=True)
        # --- END FIX ---
        
        logger.info(f"Overwriting secrets file at: {config_file_path}")
        
        # Overwrite the secrets.toml in that specific directory
        with open(config_file_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
            
        logger.info(f"Successfully created/overwritten {config_file_path}")
        return config_dict # Return the dictionary
    except Exception as e:
        logger.error(f"Failed to write secrets.toml: {e}")
        return None

# --- STEP 1: RUN THE CONFIG CREATION FIRST ---
# This code runs *immediately*
cloud_config = create_secrets_file_at_startup()
if not cloud_config:
    logger.critical("Could not create config from secrets. Exiting.")
    sys.exit(1)

# --- STEP 2: NOW, IMPORT THE OTHER MODULES ---
# By importing here, database_manager will read the
# secrets.toml file we *just* created.
import networkx as nx
import database_manager 

# --- STEP 3: DEFINE AND RUN THE MAIN LOGIC ---
def main():
    logger.info("Starting pre-computation job...")
    
    # Initialize DatabaseManager
    # We pass the config_dict to satisfy the TypeError from earlier
    logger.info("Initializing DatabaseManager...")
    db_manager = database_manager.DatabaseManager(cloud_config) 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. Check credentials in GitHub secrets.")
        sys.exit(1) # Exit with an error

    logger.info("✅ Connected to Neo4j. Fetching full graph...")
    
    # This is the slow part (can take a long time)
    graph = db_manager.get_graph_from_db(weight_threshold=0.1)
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty. Saving an empty graph file.")
    else:
        logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")

    # Save the computed graph to a file
    output_file = "financial_graph.gml"
    nx.write_gml(graph, output_file)
    
    logger.info(f"Successfully saved graph to {output_file}")
    
    # Clean up the config file (optional, but good practice)
    try:
        config_file_path = ".streamlit/secrets.toml"
        os.remove(config_file_path)
        logger.info(f"Cleaned up temporary config file at {config_file_path}")
    except Exception as e:
        logger.warning(f"Could not remove {config_file_path}: {e}")

if __name__ == "__main__":
    main()