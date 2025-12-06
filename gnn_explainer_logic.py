import torch
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
from dowhy import CausalModel
import networkx as nx

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

def extract_narrative_triples(G_expl, ticker_map, event_map=None):
    """
    Converts the explanation subgraph into text triples for the LLM.
    NOW SUPPORTS: resolving 'Event_123' -> 'Fed Rate Hike (Event)'
    """
    triples = []
    # If no map provided, use empty dict to prevent errors
    if event_map is None: 
        event_map = {}
    
    for u, v, data in G_expl.edges(data=True):
        weight = data.get('weight', 0)
        
        # Filter noise (ignore very weak edges)
        if weight < 0.05: 
            continue

        # --- Helper to Resolve Names ---
        def resolve(node_id):
            node_str = str(node_id)
            
            # 1. Resolve Company Tickers (Company_4 -> AAPL)
            if "Company_" in node_str:
                try:
                    # Extract index "Company_4" -> 4
                    idx = int(node_str.split('_')[1])
                    if 0 <= idx < len(ticker_map):
                        return f"{ticker_map[idx]} (Company)"
                except: 
                    pass
            
            # 2. Resolve Event Descriptions (Event_1922 -> "Tech Selloff")
            if "Event_" in node_str:
                # Try to find the description in the map using the full string
                if node_str in event_map:
                    return f"'{event_map[node_str]}' (Event)"
                
                # If map fails, try to parse ID and look it up by integer
                try:
                    event_id = int(node_str.split('_')[1])
                    if event_id in event_map:
                         return f"'{event_map[event_id]}' (Event)"
                except: 
                    pass
                
            return node_str

        source = resolve(u)
        target = resolve(v)
        
        # Use descriptive verbs based on weight
        relation = "strongly influences" if weight > 0.5 else "is related to"
        
        # Format: (Source) --[influences]--> (Target)
        triple = f"- {source} {relation} {target} (Importance: {weight:.2f})"
        triples.append(triple)
        
    # Sort by importance (Highest first)
    return sorted(triples, key=lambda x: float(x.split('Importance: ')[1][:-1]), reverse=True)

class CausalValidator:
    """
    Handles Counterfactual Reasoning using DoWhy.
    This moves beyond correlation (GNN weights) to causation (Do-Calculus).
    """
    def __init__(self, historical_df, graph_nx):
        """
        historical_df: Pandas DataFrame containing historical data for nodes (e.g., columns for 'Interest Rates', 'AAPL', etc.)
        graph_nx: A NetworkX DiGraph representing the causal DAG (Directed Acyclic Graph).
        """
        self.data = historical_df
        self.graph = graph_nx

    def run_counterfactual(self, treatment_node, outcome_node, perturbation=0.05):
        """
        Simulates a 'What-If' scenario using DoWhy.
        """
        # 1. Initialize Causal Model
        model = CausalModel(
            data=self.data,
            treatment=treatment_node,
            outcome=outcome_node,
            graph=self.graph
        )

        # 2. Identify the Causal Effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 3. Estimate the Effect (Linear Regression)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        # 4. Compute the Counterfactual (The "What-If")
        predicted_change = estimate.value * perturbation

        # 5. Robustness Check (Refutation)
        refute = model.refute_estimate(
            identified_estimand, 
            estimate, 
            method_name="random_common_cause"
        )
        
        # --- FIX STARTS HERE ---
        # The error "dict object has no attribute p_value" happened here.
        # We now safely extract the p-value whether it's in a dict or an object.
        p_value = None
        
        if hasattr(refute, "refutation_result") and isinstance(refute.refutation_result, dict):
            # If it's a dictionary (causing your error), use ["p_value"]
            p_value = refute.refutation_result.get("p_value")
        elif hasattr(refute, "refutation_result") and hasattr(refute.refutation_result, "p_value"):
            # If it's an object, use .p_value
            p_value = refute.refutation_result.p_value
        else:
            # Fallback: sometimes the refute object itself holds the p_value
            p_value = getattr(refute, "p_value", None)
            
        # Interpretation: 
        # For "Random Common Cause", a HIGH p-value (>0.05) is GOOD.
        # It means adding random noise did NOT significantly change our estimate (so it's robust).
        is_robust = (p_value > 0.05) if p_value is not None else False
        # --- FIX ENDS HERE ---

        return {
            "treatment": treatment_node,
            "outcome": outcome_node,
            "perturbation_amount": perturbation,
            "base_coefficient": estimate.value,
            "predicted_impact": predicted_change,
            "validity_p_value": p_value,
            "is_statistically_significant": is_robust 
        }

def convert_gnn_to_causal_graph(edge_index, node_map):
    """
    Helper to convert PyTorch Geometric edge_index to the NetworkX format DoWhy needs.
    """
    G = nx.DiGraph()
    
    # Assuming edge_index is standard [2, num_edges]
    sources = edge_index[0].cpu().numpy()
    targets = edge_index[1].cpu().numpy()
    
    for u, v in zip(sources, targets):
        u_name = node_map.get(u, f"Node_{u}")
        v_name = node_map.get(v, f"Node_{v}")
        G.add_edge(u_name, v_name)
        
    return G