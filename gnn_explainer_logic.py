import torch
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer

def setup_explainer(model):
    """
    Configures the GNNExplainer for a Heterogeneous Model.
    """
    # Freeze parameters to ensure we explain the model as-is
    for param in model.parameters():
        param.requires_grad = False

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',  # Your model returns logits
        ),
    )
    return explainer

def explain_prediction(explainer, data, target_node_idx):
    """
    Runs optimization to find the subgraph that contributed to the prediction.
    Returns: A NetworkX DiGraph of the 'Why'.
    """
    # 1. Run the explanation
    explanation = explainer(
        data.x_dict, 
        data.edge_index_dict, 
        index=target_node_idx, 
        target=None
    )

    # 2. Build the visual graph
    G_expl = nx.DiGraph()
    target_name = f"Company_{target_node_idx}"
    G_expl.add_node(target_name, type="Target", color="red", label="TARGET")

    # 3. Filter for important edges (>50% contribution)
    threshold = 0.5 
    edge_mask_dict = explanation.edge_mask_dict
    
    for edge_type, mask in edge_mask_dict.items():
        src_type, rel, dst_type = edge_type
        
        # Get indices of important edges
        important_indices = torch.where(mask > threshold)[0]
        
        if len(important_indices) > 0:
            src_nodes = data.edge_index_dict[edge_type][0][important_indices]
            dst_nodes = data.edge_index_dict[edge_type][1][important_indices]
            
            for i in range(len(src_nodes)):
                u_idx = src_nodes[i].item()
                v_idx = dst_nodes[i].item()
                
                # Naming convention: Type_Index
                u = f"{src_type}_{u_idx}"
                v = f"{dst_type}_{v_idx}"
                
                # Add nodes
                if u not in G_expl: G_expl.add_node(u, type=src_type)
                if v not in G_expl: G_expl.add_node(v, type=dst_type)
                
                # Add edge with weight
                weight = mask[important_indices[i]].item()
                G_expl.add_edge(u, v, label=rel, weight=f"{weight:.2f}")

    return G_expl