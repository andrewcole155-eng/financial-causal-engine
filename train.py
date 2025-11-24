import json
import logging
import torch
import torch.nn.functional as F
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any
from torch_geometric.data import HeteroData

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

def add_unique_company_features(data: HeteroData) -> HeteroData:
    """
    SOLUTION B: Inject Unique Random Features.
    
    This assigns a unique 16-dim random embedding to every company.
    It forces the GNN to treat every company as a unique entity, ensuring
    variance in the final risk scores even for disconnected nodes.
    """
    logger.info("--- SOLUTION B: Injecting Unique Random Features for Companies ---")
    num_companies = data['Company'].num_nodes
    
    # Generate random noise (Unique ID simulation)
    # Shape: [num_companies, 16]
    embedding_dim = 16
    unique_features = torch.randn(num_companies, embedding_dim)
    
    # Overwrite existing features (or create new ones if None)
    data['Company'].x = unique_features
    logger.info(f"Assigned random feature vectors of shape {unique_features.shape}")
    
    return data

def create_ground_truth_labels(data: HeteroData) -> HeteroData:
    """
    Creates "official" heuristic-based labels for 'Company' nodes.
    """
    logger.info("--- Creating Ground Truth Labels ---")
    
    num_companies = data['Company'].num_nodes
    
    # 1. Default all companies to Label 0 (Low Risk)
    labels = torch.zeros(num_companies, dtype=torch.long)
    
    # 2. Get event scores
    if data['Event'].num_nodes > 0:
        # Check if features exist to avoid crash on empty event features
        if data['Event'].x is not None:
            event_scores = data['Event'].x.squeeze()
            
            # 3. Get edge connections
            if ('Company', 'had_event', 'Event') in data.edge_index_dict:
                edges = data['Company', 'had_event', 'Event'].edge_index
                
                # Handle single-event dimension edge case
                if event_scores.dim() == 0:
                    event_scores = event_scores.unsqueeze(0)
                    
                medium_risk_event_indices = torch.where(
                    (event_scores < -0.3) & (event_scores >= -0.8)
                )[0]
                high_risk_event_indices = torch.where(event_scores < -0.8)[0]

                # 5. Map back to companies
                medium_risk_event_set = set(medium_risk_event_indices.tolist())
                high_risk_event_set = set(high_risk_event_indices.tolist())

                medium_risk_companies = set()
                high_risk_companies = set()

                if edges.numel() > 0:
                    for i in range(edges.shape[1]):
                        company_idx = edges[0, i].item()
                        event_idx = edges[1, i].item()
                        
                        if event_idx in high_risk_event_set:
                            high_risk_companies.add(company_idx)
                        elif event_idx in medium_risk_event_set:
                            medium_risk_companies.add(company_idx)

                # 6. Apply labels
                for company_idx in medium_risk_companies:
                    labels[company_idx] = 1
                    
                for company_idx in high_risk_companies:
                    labels[company_idx] = 2
            else:
                logger.warning("No 'had_event' edges found. Skipping label logic based on edges.")
        else:
             logger.warning("Event nodes have no features (.x is None). Skipping label logic.")

    data['Company'].y = labels
    
    logger.info(f"Label Distribution: "
                f"Low(0): {(labels == 0).sum()}, "
                f"Med(1): {(labels == 1).sum()}, "
                f"High(2): {(labels == 2).sum()}")
    
    return data

def create_train_val_test_masks(data: HeteroData) -> HeteroData:
    """Splits company nodes into train, validation, and test sets."""
    logger.info("--- Creating Train/Val/Test Masks ---")
    num_companies = data['Company'].num_nodes
    
    perm = torch.randperm(num_companies)
    
    train_pct = 0.7
    val_pct = 0.15
    
    train_end = int(train_pct * num_companies)
    val_end = int((train_pct + val_pct) * num_companies)
    
    train_indices = perm[:train_end]
    val_indices = perm[train_end:val_end]
    test_indices = perm[val_end:]
    
    train_mask = torch.zeros(num_companies, dtype=torch.bool)
    val_mask = torch.zeros(num_companies, dtype=torch.bool)
    test_mask = torch.zeros(num_companies, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    data['Company'].train_mask = train_mask
    data['Company'].val_mask = val_mask
    data['Company'].test_mask = test_mask
    
    return data

def run_training():
    """
    Main script to load data, train the GNN, save the model, AND WRITE PREDICTIONS TO DB.
    """
    logger.info("🚀 Starting GNN training script...")
    
    # 1. Load config & Data
    config = load_config()
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return
    
    # --- APPLY SOLUTION B HERE ---
    # Inject unique features immediately after loading
    data = add_unique_company_features(data)
    # -----------------------------

    # 2. Prep Data
    data = create_ground_truth_labels(data)
    data = create_train_val_test_masks(data)
    
    # 3. Initialize Model
    # We use hidden_dim=32 to be compatible/larger than our input embedding of 16
    model = create_hetero_model(data, hidden_dim=32, out_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Training Loop
    logger.info("--- Starting training loop ---")
    
    for epoch in range(1, 61): # 60 epochs
        model.train()
        optimizer.zero_grad()

        # Forward pass
        company_out = model(data.x_dict, data.edge_index_dict)

        # Compute loss (Train mask only)
        loss = loss_fn(
            company_out[data['Company'].train_mask], 
            data['Company'].y[data['Company'].train_mask]
        )
        
        loss.backward()
        optimizer.step()
        
        # Validation Log
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x_dict, data.edge_index_dict)
                pred = val_out.argmax(dim=1)
                correct = (pred[data['Company'].val_mask] == data['Company'].y[data['Company'].val_mask]).sum()
                acc = int(correct) / int(data['Company'].val_mask.sum())
                logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")

    logger.info("✅ Training complete.")

    # 5. Save Model File
    torch.save(model.state_dict(), "gnn_risk_model.pth")
    
    # --- 6. CALCULATE & SAVE RISK SCORES TO NEO4J ---
    logger.info("🔮 Generating final risk scores for database update...")
    model.eval()
    with torch.no_grad():
        # Run on ALL nodes
        logits = model(data.x_dict, data.edge_index_dict)
        
        # Convert Logits -> Probabilities (Softmax)
        # shape: [num_companies, 3] -> columns are [Prob(Low), Prob(Med), Prob(High)]
        probs = F.softmax(logits, dim=1)
        
        # Calculate a single "Risk Score" (0.0 to 1.0)
        # Formula: 0.5 * Prob(Medium) + 1.0 * Prob(High)
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        
        # Send to Pipeline to write to Neo4j
        pipeline.save_predictions(risk_scores)

if __name__ == "__main__":
    run_training()