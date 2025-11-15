# graph/analysis.py

import logging
from typing import Dict, Optional, Tuple

import networkx as nx
import pandas as pd

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_impact_scores(graph: nx.DiGraph, start_node: str, event_magnitude: float = 1.0) -> Dict[str, float]:
    """
    Calculates the propagated impact of an event starting from a single node.

    This function traverses the graph outwards from the start_node, calculating
    the potential impact on other nodes based on edge weights. It correctly
    inverts the impact for 'competitor' relationships.

    Args:
        graph: The NetworkX DiGraph representing the financial relationships.
        start_node: The ticker symbol of the company where the event originated.
        event_magnitude: The initial score of the event (e.g., sentiment score).

    Returns:
        A dictionary of impacted ticker symbols and their calculated impact scores,
        sorted in descending order of absolute impact.
    """
    if start_node not in graph:
        logger.warning(f"Start node '{start_node}' not found in the graph. Cannot calculate impact.")
        return {}

    impact_scores: Dict[str, float] = {node: 0.0 for node in graph.nodes}
    impact_scores[start_node] = event_magnitude

    # Use a queue for a Breadth-First Search (BFS) traversal
    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)

        for neighbor in graph.successors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor, default={})
            weight = edge_data.get('weight', 0.0)
            relationship_type = edge_data.get('type', 'dependency')

            # Calculate the impact passed to the neighbor
            propagated_impact = impact_scores[current_node] * weight

            # Invert the impact for competitor relationships
            if relationship_type == 'competitor':
                propagated_impact *= -1

            # Update the neighbor's score only if this path has a stronger impact
            if abs(propagated_impact) > abs(impact_scores[neighbor]):
                impact_scores[neighbor] = propagated_impact

            # Add the neighbor to the queue to continue the traversal
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Filter out the start node and nodes with no impact
    final_impacts = {
        node: score for node, score in impact_scores.items() if score != 0.0 and node != start_node
    }

    # Sort by the absolute value of the impact score to find the most affected companies
    return dict(sorted(final_impacts.items(), key=lambda item: abs(item[1]), reverse=True))


def analyze_stock_performance(
    stock_df: Optional[pd.DataFrame],
    benchmark_df: Optional[pd.DataFrame]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Analyzes a stock's performance against a benchmark over a given period.

    Calculates the total return for both the stock and the benchmark, and
    then computes the alpha (the stock's outperformance or underperformance).

    Args:
        stock_df: A Pandas DataFrame with historical price data for the stock.
                  Must contain a 'close' column.
        benchmark_df: A Pandas DataFrame with historical price data for the benchmark (e.g., SPY).
                      Must contain a 'close' column.

    Returns:
        A tuple containing (stock_return, benchmark_return, alpha). Returns
        (None, None, None) if input data is invalid.
    """
    if stock_df is None or benchmark_df is None or stock_df.empty or benchmark_df.empty:
        logger.warning("Stock or benchmark DataFrame is empty or None. Cannot analyze performance.")
        return None, None, None

    try:
        # Calculate the percentage return over the period using the first and last closing prices
        stock_return = (stock_df['close'].iloc[-1] - stock_df['close'].iloc[0]) / stock_df['close'].iloc[0]
        benchmark_return = (benchmark_df['close'].iloc[-1] - benchmark_df['close'].iloc[0]) / benchmark_df['close'].iloc[0]

        # Calculate alpha (the stock's performance relative to the benchmark)
        alpha = stock_return - benchmark_return

        return stock_return, benchmark_return, alpha
    except (IndexError, KeyError) as e:
        logger.error(f"Could not calculate performance due to missing data or incorrect DataFrame structure: {e}")
        return None, None, None