import logging
import os
import sys
import json
import networkx as nx
import database_manager

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_local_config():
    """
    Loads config.json from the same directory as the script.
    Mimics the logic in macro_ingest.py which we know works.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Error reading config.json: {e}")
        return {}

def main():
    logger.info("Starting pre-computation job...")
    
    # 1. Attempt to load from Environment Variables (Cloud/Docker)
    neo4j_uri = os.environ.get('NEO4J_URI')
    neo4j_user = os.environ.get('NEO4J_USER')
    neo4j_password = os.environ.get('NEO4J_PASSWORD')

    config = {}

    if neo4j_uri and neo4j_user and neo4j_password:
        logger.info("Using credentials from Environment Variables.")
        config = {
            "neo4j": {
                "uri": neo4j_uri,
                "user": neo4j_user,
                "password": neo4j_password
            }
        }
    else:
        # 2. Fallback to config.json (Local Development)
        logger.info("Environment variables missing. Loading config.json...")
        config = load_local_config()

    # 3. Validation
    if not config:
        logger.critical("❌ CRITICAL: No configuration found (Env Vars missing and config.json failed).")
        sys.exit(1)

    # Initialize DatabaseManager
    # We pass the config dict directly, letting DatabaseManager handle the parsing logic
    # just like it does in macro_ingest.py
    logger.info("Initializing DatabaseManager...")
    db_manager = database_manager.DatabaseManager(config) 
    
    if not db_manager.is_connected():
        logger.error("Could not connect to Neo4j. Check your config.json or network.")
        sys.exit(1)

    logger.info("✅ Connected to Neo4j. Fetching full graph...")
    
    # Fetch the graph (The weight_threshold=0.1 allows our 0.9 Macro edges to pass)
    graph = db_manager.get_graph_from_db(weight_threshold=0.1)
    
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty. Saving an empty graph file.")
    else:
        logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")

        # ==============================================================================
        # --- GLOBAL STATE DIAGNOSTICS ---
        # ==============================================================================
        macro_count = 0
        global_edge_count = 0
        
        # 1. Count Macro Nodes
        for node_id, data in graph.nodes(data=True):
            # Check for boolean True or string "True" just in case
            if data.get('is_macro') in [True, "true", "True"] or data.get('sector') == 'Macro':
                macro_count += 1
                
        # 2. Count Global Edges
        for u, v, data in graph.edges(data=True):
            if data.get('mechanism') == 'Global Macro Influence':
                global_edge_count += 1
        
        logger.info("📊 GLOBAL STATE DIAGNOSTICS:")
        logger.info(f"   -> Super-Nodes Found:    {macro_count} (Should be ~5)")
        logger.info(f"   -> Global Influence Edges: {global_edge_count}")
        
        if macro_count == 0:
            logger.warning("⚠️  WARNING: No Macro Nodes found! Did you run macro_ingest.py?")
        elif global_edge_count == 0:
            logger.warning("⚠️  WARNING: Macro Nodes exist but have no edges! Check connection logic.")
        else:
            logger.info("✅ Global State topology is healthy.")
        # ==============================================================================

    # Save the computed graph to a file
    output_file = "financial_graph.gml"
    nx.write_gml(graph, output_file)
    
    logger.info(f"Successfully saved graph to {output_file}")

if __name__ == "__main__":
    main()