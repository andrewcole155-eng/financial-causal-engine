# predict.py

import json
import logging
import torch
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any
from py2neo import Graph
import os

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Loads config.json from the same directory."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config.json: {e}")
        return {}

def get_neo4j_graph(config: Dict[str, Any]) -> Graph | None:
    """Creates a new py2neo Graph connection."""
    try:
        neo4j_uri = os.getenv("NEO4J_URI", config.get("neo4j_uri"))
        neo4j_user = os.getenv("NEO4J_USER", config.get("neo4j_user"))
        neo4j_password = os.getenv("NEO4J_PASSWORD", config.get("neo4j_password"))
        
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j connection details not found.")
            
        graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        graph.run("MATCH (n) RETURN count(n)")
        logger.info("✅ Neo4j connection successful for writing predictions.")
        return graph
    except Exception as e:
        logger.critical(f"❌ Failed to connect to Neo4j: {e}")
        return None

def run_inference():
    """
    Loads the trained GNN model and uses it to predict risk
    for all companies, then writes those predictions back to Neo4j.
    """
    logger.info("🚀 Starting GNN inference script...")
    
    config = load_config()
    if not config:
        return

    # --- 1. Load Graph Data ---
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return
    logger.info("Graph data loaded successfully.")

    # --- 2. Load Trained Model ---
    model_path = "gnn_risk_model.pth"
    try:
        # We need to initialize the model architecture first
        model = create_hetero_model(data, hidden_dim=64, out_dim=3)
        
        # Now, load the saved weights (the "brain")
        model.load_state_dict(torch.load(model_path))
        
        # Set model to "evaluation" mode (disables dropout, etc.)
        model.eval()
        logger.info(f"Successfully loaded trained model from {model_path}")
    except FileNotFoundError:
        logger.error(f"FATAL: Model file not found at {model_path}. Please run train.py first.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- 3. Run Inference ---
    logger.info("Running model to get predictions for all companies...")
    with torch.no_grad(): # Disable gradient calculation for speed
        
        # Run the *entire graph* through the model
        all_company_predictions = model(data.x_dict, data.edge_index_dict)
        
        # Get the final predicted class (0, 1, or 2) for each company
        # .argmax(dim=1) finds the index with the highest score
        predicted_labels = all_company_predictions.argmax(dim=1)
        
        # Convert from tensor to a simple Python list
        predictions_list = predicted_labels.cpu().numpy().tolist()
        logger.info(f"Predictions generated for {len(predictions_list)} companies.")

    # --- 4. Write Predictions back to Neo4j ---
    # We need the original Neo4j node IDs to write back to the DB.
    # We can re-query them (simpler) or save the map from the pipeline.
    # Let's re-query for simplicity.
    
    neo4j_graph = get_neo4j_graph(config)
    if not neo4j_graph:
        logger.error("Could not connect to Neo4j to write predictions.")
        return
        
    logger.info("Writing predictions back to Neo4j...")
    
    # This query gets all companies IN THE SAME ORDER as the GNN pipeline
    # This is CRITICAL for matching predictions to the right company.
    company_query = """
    MATCH (c:Company)
    RETURN c.ticker AS ticker, id(c) AS neo4j_id
    """
    company_nodes = neo4j_graph.run(company_query).data()
    
    # Create a list of dictionaries for the batch update
    updates = []
    for i, node in enumerate(company_nodes):
        updates.append({
            "ticker": node['ticker'],
            "predicted_risk": predictions_list[i]
        })

    # Use UNWIND for a fast, batch-update query
    write_query = """
    UNWIND $updates AS row
    MATCH (c:Company {ticker: row.ticker})
    SET c.predicted_risk = row.predicted_risk
    """
    
    try:
        neo4j_graph.run(write_query, updates=updates)
        logger.info(f"✅ Successfully updated {len(updates)} Company nodes in Neo4j with 'predicted_risk'.")
    except Exception as e:
        logger.error(f"Failed to write predictions to Neo4j: {e}")

if __name__ == "__main__":
    run_inference()