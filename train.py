import json
import logging
import torch
import torch.nn.functional as F
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any
from torch_geometric.data import HeteroData
import os
import sys 

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION HELPER (Same as precompute_graph.py) ---
def load_local_config():
    """Loads config.json from the same directory as the script."""
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

# --- HELPER FUNCTIONS (No Changes Here) ---
def add_unique_company_features(data: HeteroData) -> HeteroData:
    logger.info("--- SOLUTION B: Injecting Unique Random Features for Companies ---")
    torch.manual_seed(42)
    num_companies = data['Company'].num_nodes
    embedding_dim = 16
    unique_features = torch.randn(num_companies, embedding_dim)
    data['Company'].x = unique_features
    logger.info(f"Assigned random feature vectors of shape {unique_features.shape}")
    return data

def create_ground_truth_labels(data: HeteroData) -> HeteroData:
    logger.info("--- Creating Ground Truth Labels ---")
    num_companies = data['Company'].num_nodes
    labels = torch.zeros(num_companies, dtype=torch.long)
    
    if data['Event'].num_nodes > 0 and data['Event'].x is not None:
        event_scores = data['Event'].x.squeeze()
        if ('Company', 'had_event', 'Event') in data.edge_index_dict:
            edges = data['Company', 'had_event', 'Event'].edge_index
            if event_scores.dim() == 0: event_scores = event_scores.unsqueeze(0)
            
            medium_risk_event_indices = torch.where((event_scores < -0.3) & (event_scores >= -0.8))[0]
            high_risk_event_indices = torch.where(event_scores < -0.8)[0]

            medium_risk_event_set = set(medium_risk_event_indices.tolist())
            high_risk_event_set = set(high_risk_event_indices.tolist())

            for i in range(edges.shape[1]):
                company_idx = edges[0, i].item()
                event_idx = edges[1, i].item()
                if event_idx in high_risk_event_set: labels[company_idx] = 2
                elif event_idx in medium_risk_event_set: labels[company_idx] = 1
    
    data['Company'].y = labels
    return data

def create_train_val_test_masks(data: HeteroData) -> HeteroData:
    logger.info("--- Creating Train/Val/Test Masks ---")
    num_companies = data['Company'].num_nodes
    perm = torch.randperm(num_companies)
    
    train_end = int(0.7 * num_companies)
    val_end = int(0.85 * num_companies)
    
    train_mask = torch.zeros(num_companies, dtype=torch.bool)
    val_mask = torch.zeros(num_companies, dtype=torch.bool)
    test_mask = torch.zeros(num_companies, dtype=torch.bool)
    
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True
    
    data['Company'].train_mask = train_mask
    data['Company'].val_mask = val_mask
    data['Company'].test_mask = test_mask
    return data

# --- MAIN EXECUTION ---
def run_training():
    logger.info("🚀 Starting GNN training script...")
    
    # ==========================================================================
    # 1. ROBUST CONFIGURATION LOAD (Fixes the KeyError)
    # ==========================================================================
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
        logger.info("Environment variables missing. Loading config.json...")
        loaded_config = load_local_config()
        
        # Ensure the structure matches what GNNPipeline expects: {'neo4j': {...}}
        if "neo4j" in loaded_config:
            config = loaded_config
        else:
            # If flat config, wrap it
            config = {"neo4j": loaded_config}

    # Final Validation
    if not config or "neo4j" not in config or not config["neo4j"]:
        logger.critical("❌ CRITICAL: No valid configuration found. Exiting.")
        sys.exit(1)
    # ==========================================================================

    # 2. Load Data Pipeline
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return
    
    # 3. Apply Enhancements
    data = add_unique_company_features(data)
    data = create_ground_truth_labels(data)
    data = create_train_val_test_masks(data)
    
    # 4. Initialize Model (Using 32 hidden dims to capture the 16 random + structural features)
    model = create_hetero_model(data, hidden_dim=32, out_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. Training Loop
    logger.info("--- Starting training loop ---")
    
    for epoch in range(1, 61): 
        model.train()
        optimizer.zero_grad()

        # Forward pass (Now includes GAT Attention on Macro nodes!)
        company_out = model(data.x_dict, data.edge_index_dict)

        # Compute loss (Train mask only)
        loss = loss_fn(
            company_out[data['Company'].train_mask], 
            data['Company'].y[data['Company'].train_mask]
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x_dict, data.edge_index_dict)
                pred = val_out.argmax(dim=1)
                correct = (pred[data['Company'].val_mask] == data['Company'].y[data['Company'].val_mask]).sum()
                total_val = int(data['Company'].val_mask.sum())
                acc = int(correct) / total_val if total_val > 0 else 0.0
                logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

    logger.info("✅ Training complete.")

    # 6. Save Model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "gnn_risk_model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"💾 Model saved to: {save_path}")
    
    # 7. Calculate & Save Risk Scores
    logger.info("🔮 Generating final risk scores for database update...")
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(logits, dim=1)
        # Weighted risk score: 0.5 * Medium_Prob + 1.0 * High_Prob
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        
        pipeline.save_predictions(risk_scores)

if __name__ == "__main__":
    run_training()