import logging
import os
import sys
import networkx as nx
import database_manager  # This import is now safe

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting pre-computation job...")
    
    # --- THIS IS THE FIX ---
    # Load secrets *directly* from the environment.
    # The 'env:' block in warmup.yml makes these available.
    logger.info("Loading secrets directly from environment...")
    neo4j_uri = os.environ.get('NEO4J_URI')
    neo4j_user = os.environ.get('NEO4J_USER')
    neo4j_password = os.environ.get('NEO4J_PASSWORD')

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        logger.critical("Missing one or more Neo4j credentials in environment variables.")
        sys.exit(1) # Exit with an error

    # Create the config dictionary to pass to the DatabaseManager
    # This matches the structure your new DatabaseManager expects
    cloud_config = {
        "neo4j": {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_password
        }
    }
    # --- END FIX ---

    # Initialize DatabaseManager
    logger.info("Initializing DatabaseManager...")
    db_manager = database_manager.DatabaseManager(cloud_config) 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. This is now likely a credential or firewall issue.")
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
    
    # (No config file to clean up)

if __name__ == "__main__":
    main()