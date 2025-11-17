import json
import logging
import networkx as nx
from database_manager import DatabaseManager

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file: str = "config.json") -> dict:
    """Loads all configurations from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fatal: Error loading configuration file '{config_file}': {e}")
        return {}

def main():
    logger.info("Starting pre-computation job...")
    
    # Load secrets (this assumes you run this in GitHub Actions
    # and have secrets available as environment variables)
    import os
    
    # You MUST set these as secrets in your GitHub repository
    # (Settings -> Secrets and variables -> Actions)
    NEO4J_URI = os.environ.get('NEO4J_URI')
    NEO4J_USER = os.environ.get('NEO4J_USER')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        logger.error("Missing Neo4j credentials in environment variables.")
        return

    cloud_config = {
        "neo4j": {
            "uri": NEO4J_URI,
            "user": NEO4J_USER,
            "password": NEO4J_PASSWORD
        }
    }
    
    db_manager = DatabaseManager(cloud_config)
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j.")
        return

    logger.info("Connected to Neo4j. Fetching full graph...")
    
    # This is the slow part (can take 1 hour)
    graph = db_manager.get_graph_from_db(weight_threshold=0.1)
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty. Saving an empty graph file.")
    else:
        logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")

    # Save the computed graph to a file
    # GML is a standard format for saving NetworkX graphs
    output_file = "financial_graph.gml"
    nx.write_gml(graph, output_file)
    
    logger.info(f"Successfully saved graph to {output_file}")

if __name__ == "__main__":
    main()