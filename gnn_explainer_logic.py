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
            return_type='raw',
        ),
    )
    return explainer

def explain_prediction(explainer, data, target_node_idx, top_k=20):
    """
    Runs optimization and returns ONLY the top_k most important edges.
    This prevents the "Hairball" visualization.
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

    # 3. Collect ALL edges with their importance scores
    all_edges = []
    
    edge_mask_dict = explanation.edge_mask_dict
    
    for edge_type, mask in edge_mask_dict.items():
        src_type, rel, dst_type = edge_type
        
        # Get indices where mask is non-zero
        indices = torch.where(mask > 0.01)[0] # minimal filter
        
        if len(indices) > 0:
            src_nodes = data.edge_index_dict[edge_type][0][indices]
            dst_nodes = data.edge_index_dict[edge_type][1][indices]
            weights = mask[indices]
            
            for i in range(len(src_nodes)):
                edge_info = {
                    'u': f"{src_type}_{src_nodes[i].item()}",
                    'v': f"{dst_type}_{dst_nodes[i].item()}",
                    'u_type': src_type,
                    'v_type': dst_type,
                    'rel': rel,
                    'weight': weights[i].item()
                }
                all_edges.append(edge_info)

    # 4. Sort by importance (Descending) and slice Top K
    all_edges.sort(key=lambda x: x['weight'], reverse=True)
    top_edges = all_edges[:top_k]

    # 5. Construct the clean Graph
    for e in top_edges:
        # Add nodes if missing
        if e['u'] not in G_expl: G_expl.add_node(e['u'], type=e['u_type'])
        if e['v'] not in G_expl: G_expl.add_node(e['v'], type=e['v_type'])
        
        # Add Weighted Edge
        G_expl.add_edge(e['u'], e['v'], label=e['rel'], weight=e['weight'])

    return G_expl

def extract_narrative_triples(G_expl, ticker_map):
    """
    Converts the explanation subgraph into text triples for the LLM.
    e.g. "Company_4" -> "Apple"
    """
    triples = []
    
    for u, v, data in G_expl.edges(data=True):
        weight = data.get('weight', 0)
        
        # Filter out weak connections to reduce noise for the LLM
        if weight < 0.05: 
            continue

        # --- Helper to Resolve Names ---
        def resolve(node_id):
            if isinstance(node_id, str) and "Company_" in node_id:
                try:
                    idx = int(node_id.split('_')[1])
                    if 0 <= idx < len(ticker_map):
                        return ticker_map[idx] # Return Ticker (e.g., AAPL)
                except:
                    pass
            # Fallback for Event nodes or raw strings
            return str(node_id)

        source = resolve(u)
        target = resolve(v)
        
        # Determine relationship verb
        # In the future, you can pull 'mechanism' from the edge data if available
        relation = "strongly influences" if weight > 0.5 else "is related to"
        
        # Format: (Source) --[influences]--> (Target)
        triple = f"- {source} {relation} {target} (Importance: {weight:.2f})"
        triples.append(triple)
        
    # Sort by importance so the LLM sees the biggest drivers first
    return sorted(triples, key=lambda x: float(x.split('Importance: ')[1][:-1]), reverse=True)