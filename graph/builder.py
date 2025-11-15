# graph/builder.py

import json
import logging
import time
import os
import glob
from typing import List, Dict, Any

import networkx as nx
import nltk
from sentence_transformers import SentenceTransformer, util

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_nodes_with_api_data(tickers: List[str], client: Any) -> nx.DiGraph:
    """
    Builds a NetworkX graph with company nodes fetched from the Polygon API.

    Args:
        tickers: A list of ticker symbols to create nodes for.
        client: An initialized polygon.RESTClient instance.

    Returns:
        A NetworkX DiGraph populated with nodes and their metadata.
    """
    graph = nx.DiGraph()
    if not client:
        logger.error("API client is missing. Cannot build nodes.")
        return graph

    logger.info("Building nodes with Polygon API client...")
    for ticker in tickers:
        try:
            # Note: client is of type polygon.RESTClient
            resp = client.get_ticker_details(ticker)
            graph.add_node(
                ticker,
                name=getattr(resp, 'name', 'N/A'),
                sector=getattr(resp, 'sic_description', 'N/A'),
                market_cap=getattr(resp, 'market_cap', 0)
            )
            logger.info(f"✅ Added node for {ticker}: {getattr(resp, 'name', 'N/A')}")
        except Exception as e:
            logger.error(f"❌ Could not fetch data for {ticker}. Error: {e}")
            graph.add_node(ticker, name=f"{ticker} (Data Fetch Failed)", sector="Unknown", market_cap=0)
        # Respect API rate limits (5 requests per minute for free tier)
        time.sleep(13)
    return graph


def add_manual_relationships(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Adds manually defined, high-conviction edges to the graph.

    Args:
        graph: The NetworkX DiGraph to add edges to.

    Returns:
        The graph with added manual relationships.
    """
    if not isinstance(graph, nx.Graph):
        logger.error("A valid graph object was not provided.")
        return graph

    logger.info("🔗 Adding manually defined relationships...")
    edges = [
        ("KO", "KR", {"label": "sells_products_through", "type": "dependency", "weight": 0.9}),
        ("OXY", "KO", {"label": "impacts_mfg_logistics_costs", "type": "dependency", "weight": 0.6}),
        ("OXY", "KR", {"label": "impacts_logistics_costs", "type": "dependency", "weight": 0.5}),
        ("OXY", "INTC", {"label": "impacts_fab_energy_costs", "type": "dependency", "weight": 0.7}),
        ("INTC", "KR", {"label": "enables_logistics_systems", "type": "dependency", "weight": 0.5}),
        ("INTC", "KO", {"label": "enables_global_operations", "type": "dependency", "weight": 0.5}),
        ("INTC", "OXY", {"label": "enables_seismic_analysis", "type": "dependency", "weight": 0.6}),
        ("INTC", "AMD", {"label": "competes_with", "type": "competitor", "weight": 0.8}),
        ("AMD", "INTC", {"label": "competes_with", "type": "competitor", "weight": 0.8}),
    ]
    for u, v, attrs in edges:
        # Ensure nodes exist before adding an edge to prevent errors
        if graph.has_node(u) and graph.has_node(v):
            graph.add_edge(u, v, **attrs)
    logger.info("✅ Manual relationships have been added to the graph.")
    return graph


def download_sec_filings(tickers: List[str], downloader_instance: Any) -> None:
    """
    Downloads the most recent 10-K filing for a list of tickers.

    Args:
        tickers: A list of ticker symbols.
        downloader_instance: An initialized sec_edgar_downloader.Downloader instance.
    """
    logger.info("🏛️  Downloading SEC 10-K filings...")
    for ticker in tickers:
        try:
            # Download the 1 most recent 10-K filing
            downloader_instance.get("10-K", ticker, limit=1, download_details=False)
            logger.info(f"  -> Successfully downloaded 10-K for {ticker}")
        except Exception as e:
            logger.error(f"  -> Could not download 10-K for {ticker}. Error: {e}")
    logger.info("✅ All filings downloaded.")


def discover_relationships_from_filings(graph: nx.DiGraph, tickers: List[str], force_reparse: bool = False) -> nx.DiGraph:
    """
    Parses downloaded SEC filings to discover and add new relationships to the graph.
    Uses a cached JSON file for efficiency on subsequent runs.

    Args:
        graph: The NetworkX DiGraph to add new relationships to.
        tickers: The list of core tickers whose filings will be parsed.
        force_reparse: If True, ignores the cache and re-parses all filings.

    Returns:
        The graph with newly discovered relationships added.
    """
    logger.info("🔍 Discovering relationships from SEC filings...")
    cache_file = 'sec_discovered_relationships.json'

    if not force_reparse and os.path.exists(cache_file):
        logger.info("  -> Found cached relationships file. Loading...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            discovered_rels = json.load(f)

        for rel in discovered_rels:
            source, target, attrs = rel['source'], rel['target'], rel['attrs']
            if not graph.has_node(target):
                graph.add_node(target, name=target, sector="Discovered", market_cap=0)
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, **attrs)

        logger.info(f"  -> ✅ Loaded {len(discovered_rels)} relationships from cache.")
        return graph

    logger.info("  -> No cache found or re-parse forced. Starting deep semantic parsing...")
    try:
        with open('sp500_map.json', 'r', encoding='utf-8') as f:
            sp500_data = json.load(f)
    except FileNotFoundError:
        logger.error("  -> 'sp500_map.json' not found. Cannot discover new relationships.")
        return graph

    all_company_names: Dict[str, str] = {
        item['name'].lower().split(' ')[0].replace('.', ''): item['ticker']
        for item in sp500_data if item.get('name') and item['name'] != 'N/A'
    }

    model = SentenceTransformer('all-MiniLM-L6-v2')
    queries = {
        'business': "the company's primary business operations, products, services, and market strategy",
        'risks': 'risks, challenges, and uncertainties facing the company and its operations'
    }
    query_embeddings = model.encode(list(queries.values()))
    discovered_rels_to_save = []
    
    competitor_keywords = {'compete', 'competitor', 'competition', 'competitive'}
    dependency_keywords = {'customer', 'client', 'partner', 'supplier', 'agreement', 'collaboration'}

    for ticker in tickers:
        try:
            search_path = os.path.join('sec-edgar-filings', ticker, '10-K', '*', 'full-submission.txt')
            filing_paths = glob.glob(search_path)
            if not filing_paths:
                continue

            with open(filing_paths[0], 'r', encoding='utf-8') as f:
                full_text = f.read()

            paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.split()) > 50]
            if not paragraphs:
                continue

            paragraph_embeddings = model.encode(paragraphs, show_progress_bar=False)
            hits = util.semantic_search(query_embeddings, paragraph_embeddings, top_k=10)
            
            relevant_corpus_ids = {hit['corpus_id'] for hit_list in hits for hit in hit_list}
            content_to_parse = " ".join([paragraphs[i] for i in relevant_corpus_ids])

            for sentence in nltk.sent_tokenize(content_to_parse):
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
                    if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                        org_name = ' '.join(c[0] for c in chunk).lower()
                        for name, related_ticker in all_company_names.items():
                            if name in org_name and ticker != related_ticker and not graph.has_edge(ticker, related_ticker):
                                sent_lower = sentence.lower()
                                rel_type = "dependency"
                                if any(word in sent_lower for word in competitor_keywords):
                                    rel_type = "competitor"
                                
                                attrs = {"label": f"sec_{rel_type}", "type": rel_type, "weight": 0.75}
                                logger.info(f"  -> Discovered: {ticker} -> {related_ticker} (Type: {rel_type})")
                                
                                if not graph.has_node(related_ticker):
                                    graph.add_node(related_ticker, name=related_ticker, sector="Discovered", market_cap=0)
                                graph.add_edge(ticker, related_ticker, **attrs)
                                discovered_rels_to_save.append({'source': ticker, 'target': related_ticker, 'attrs': attrs})
        except Exception as e:
            logger.error(f"  -> Error processing filing for {ticker}: {e}")
            
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(discovered_rels_to_save, f, indent=4)
    logger.info(f"  -> ✅ Finished parsing and saved {len(discovered_rels_to_save)} new relationships to cache.")
    
    return graph