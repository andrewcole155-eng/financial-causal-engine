# train.py

import json
import logging
import torch
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any, Tuple
from torch_geometric.data import HeteroData

# --- Setup logging ---
# Note: Fixed a small typo in your logging format string (message.s -> message)
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

def create_ground_truth_labels(data: HeteroData) -> HeteroData:
    """
    Creates "official" heuristic-based labels for 'Company' nodes.
    
    Logic:
    - 0 (Low Risk): Default
    - 1 (Medium Risk): Has at least one event with score < -0.3
    - 2 (High Risk): Has at least one event with score < -0.8
    """
    logger.info("--- Creating Ground Truth Labels ---")
    
    num_companies = data['Company'].num_nodes
    
    # 1. Default all companies to Label 0 (Low Risk)
    labels = torch.zeros(num_companies, dtype=torch.long)
    
    # 2. Get event scores and edge connections
    # .x is [N, 1] tensor, .squeeze() makes it a [N] vector
    event_scores = data['Event'].x.squeeze()
    
    # [2, num_edges] tensor. Row 0 is company_idx, Row 1 is event_idx
    edges = data['Company', 'had_event', 'Event'].edge_index
    
    # 3. Find indices of "Medium" and "High" risk events
    medium_risk_event_indices = torch.where(
        (event_scores < -0.3) & (event_scores >= -0.8)
    )[0]
    high_risk_event_indices = torch.where(event_scores < -0.8)[0]

    # 4. Find the companies connected to these events
    # We use sets for efficient lookup
    medium_risk_event_set = set(medium_risk_event_indices.numpy())
    high_risk_event_set = set(high_risk_event_indices.numpy())

    medium_risk_companies = set()
    high_risk_companies = set()

    # Iterate over all edges
    for i in range(edges.shape[1]):
        company_idx = edges[0, i].item()
        event_idx = edges[1, i].item()
        
        if event_idx in high_risk_event_set:
            high_risk_companies.add(company_idx)
        elif event_idx in medium_risk_event_set:
            medium_risk_companies.add(company_idx)

    # 5. Apply labels. High risk (2) overrides Medium risk (1).
    for company_idx in medium_risk_companies:
        labels[company_idx] = 1
        
    for company_idx in high_risk_companies:
        labels[company_idx] = 2

    data['Company'].y = labels
    
    # --- Log the distribution of our new labels ---
    logger.info(f"Label Distribution: "
                f"Low Risk (0): {(labels == 0).sum()} companies, "
                f"Medium Risk (1): {(labels == 1).sum()} companies, "
                f"High Risk (2): {(labels == 2).sum()} companies")
    
    if (labels == 0).sum() == num_companies:
        logger.warning("No medium or high risk companies found. The GNN may not learn well.")
    
    return data

def create_train_val_test_masks(data: HeteroData) -> HeteroData:
    """
    Splits company nodes into train, validation, and test sets.
    This is CRITICAL to properly evaluate the model.
    """
    logger.info("--- Creating Train/Val/Test Masks ---")
    num_companies = data['Company'].num_nodes
    
    # Create a random permutation of indices
    perm = torch.randperm(num_companies)
    
    # Define split percentages
    train_pct = 0.7
    val_pct = 0.15
    # Test pct is (1 - train - val) = 0.15
    
    train_end = int(train_pct * num_companies)
    val_end = int((train_pct + val_pct) * num_companies)
    
    train_indices = perm[:train_end]
    val_indices = perm[train_end:val_end]
    test_indices = perm[val_end:]
    
    # Create boolean masks
    train_mask = torch.zeros(num_companies, dtype=torch.bool)
    val_mask = torch.zeros(num_companies, dtype=torch.bool)
    test_mask = torch.zeros(num_companies, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    data['Company'].train_mask = train_mask
    data['Company'].val_mask = val_mask
    data['Company'].test_mask = test_mask
    
    logger.info(f"Company nodes split: "
                f"Train: {train_mask.sum()}, "
                f"Val: {val_mask.sum()}, "
                f"Test: {test_mask.sum()}")
    
    return data

def run_training():
    """
    Main script to load data, train the GNN, and save the model.
    """
    logger.info("🚀 Starting GNN training script...")
    
    # 1. Load config
    config = load_config()
    if not config:
        return

    # 2. Get Data from GNN Pipeline
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return
    logger.info("Graph data loaded successfully.")
    
    # --- 3. CREATE GROUND TRUTH (NOW OFFICIAL!) ---
    data = create_ground_truth_labels(data)
    
    # --- 3b. CREATE DATA SPLITS ---
    data = create_train_val_test_masks(data)
    
    # --- 4. Initialize Model, Optimizer, and Loss ---
    model = create_hetero_model(data, hidden_dim=64, out_dim=3) # out_dim=3 (Low, Med, High)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- 5. Run the Training Loop ---
    logger.info("--- Starting training loop ---")
    
    for epoch in range(1, 101):
        
        # --- Training Step ---
        model.train() # Set the model to "training" mode
        optimizer.zero_grad()

        # Get the model's prediction
        company_out = model(data.x_dict, data.edge_index_dict)

        # Get the predictions *only* for Companies
        # (This line is now removed)

        # Calculate loss ONLY on the training data
        loss = loss_fn(
            company_out[data['Company'].train_mask], 
            data['Company'].y[data['Company'].train_mask]
        )
        
        loss.backward()
        optimizer.step()
        
        # --- Validation Step ---
        if epoch % 10 == 0:
            model.eval() # Set the model to "evaluation" mode
            with torch.no_grad(): # Disable gradient calculations
                
                # Run the model on the whole graph
                val_out = model(data.x_dict, data.edge_index_dict)

                # Get the model's prediction (the class with the highest score)
                pred = val_out.argmax(dim=1)

                # Check accuracy ONLY on the validation data
                correct = (
                    pred[data['Company'].val_mask] == 
                    data['Company'].y[data['Company'].val_mask]
                ).sum()
                
                total = data['Company'].val_mask.sum()
                val_acc = int(correct) / int(total)
                
                logger.info(f"Epoch: {epoch:03d}, "
                            f"Train Loss: {loss.item():.4f}, "
                            f"Validation Acc: {val_acc:.4f}")

    logger.info("✅ Training complete.")

    # --- 6. Save the Trained Model ---
    model_path = "gnn_risk_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    run_training()