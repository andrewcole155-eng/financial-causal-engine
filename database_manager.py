# ==============================================================================
# --- IMPORTS ---
# ==============================================================================
import sqlite3
import logging
from typing import Dict, Any, List, Optional
import re
import os
import networkx as nx
from py2neo import Graph 
# from neo4j import GraphDatabase 

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- DATABASE MANAGER CLASS ---
# ==============================================================================

class DatabaseManager:
    """
    Manages all database interactions, providing a clean interface for connecting to,
    writing to, and reading from SQLite (for event logs) and Neo4j (for the graph).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes connections to SQLite and the Neo4j graph database.
        """
        logger.info("🗄️ Initializing Database Manager...")
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.neo4j_graph: Optional[Graph] = None 

        # ### CLOUD EDIT: DISABLE SQLITE CONNECTION ###
        # The Streamlit Cloud filesystem is read-only.
        # We cannot create or connect to a local .db file.
        # This entire block is now commented out.
        # --- Connect to SQLite ---
        # try:
        #     db_path = '/app/database/financial_data.db'
        #     #os.makedirs(os.path.dirname(db_path), exist_ok=True)
        #     self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        #     self.sqlite_conn.row_factory = sqlite3.Row
        #     self._create_sqlite_tables()
        #     logger.info(" -> ✅ Successfully connected to SQLite and tables are ready.")
        # except sqlite3.Error as e:
        #     logger.critical(f" -> ❌ FATAL: Failed to connect to or initialize SQLite: {e}")
        #     self.sqlite_conn = None

        # --- Connect to Neo4j ---
        try:
            # ### CLOUD EDIT: FIX CONFIG MISMATCH ###
            # app.py provides config as {'neo4j': {'uri': ...}}
            # We must access these keys directly.
            neo4j_uri = config["neo4j"]["uri"]
            neo4j_user = config["neo4j"]["user"]
            neo4j_password = config["neo4j"]["password"]

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection details not found in config.")

            self.neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            # Test connection
            self.neo4j_graph.run("MATCH (n) RETURN count(n)")
            logger.info(" -> ✅ Successfully connected to Neo4j.")
            
        except Exception as e:
            logger.critical(f" -> ❌ FATAL: Failed to connect to Neo4j: {e}")
            self.neo4j_graph = None

    def is_connected(self) -> bool:
        """Checks if the connection to Neo4j is active."""
        return self.neo4j_graph is not None

    def _create_sqlite_tables(self):
        """Creates all necessary tables in the SQLite database if they don't exist."""
        # This function will now do nothing, as self.sqlite_conn is None
        if not self.sqlite_conn: return
        with self.sqlite_conn as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS companies (
                    ticker TEXT PRIMARY KEY, name TEXT, sector TEXT,
                    market_cap REAL, last_updated TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS significant_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, ticker TEXT,
                    headline TEXT, score REAL, link TEXT
                )
            ''')

    def upsert_company_nodes_batch(self, nodes_data: List[Dict[str, Any]]):
        """
        Inserts or updates a batch of company nodes in Neo4j using a single, efficient query.
        """
        if not self.is_connected() or not nodes_data:
            return

        logger.info(f" -> Upserting batch of {len(nodes_data)} company nodes into Neo4j...")
        query = """
        UNWIND $nodes_data AS node_props
        MERGE (c:Company {ticker: node_props.ticker})
        SET c.name = node_props.name,
            c.sector = node_props.sector,
            c.market_cap = node_props.market_cap
        """
        self.neo4j_graph.run(query, nodes_data=nodes_data)

    def upsert_relationship(self, source_ticker: str, target_ticker: str, rel_type: str, properties: Dict[str, Any]):
        """
        Creates or updates a single relationship between two companies in Neo4j.
        """
        if not self.is_connected(): return

        rel_type_sanitized = re.sub(r'[^A-Z0-9_]', '', rel_type.upper())
        if not rel_type_sanitized:
            logger.warning(f"Skipping invalid relationship type: {rel_type}")
            return

        query = f"""
        MATCH (source:Company {{ticker: $source_ticker}})
        MATCH (target:Company {{ticker: $target_ticker}})
        MERGE (source)-[r:{rel_type_sanitized}]->(target)
        SET r += $properties
        """
        try:
            self.neo4j_graph.run(query, source_ticker=source_ticker, target_ticker=target_ticker, properties=properties)
        except Exception as e:
            logger.error(f"Failed to upsert relationship {source_ticker}->{target_ticker}: {e}")

    def _add_event_node_to_graph(self, ticker: str, headline: str, score: float, link: str, timestamp: str):
        """
        (Private) Creates an Event node in Neo4j and links it to a Company.
        """
        if not self.is_connected(): return

        query = """
        MATCH (c:Company {ticker: $ticker})
        MERGE (e:Event {link: $link, timestamp: $timestamp})
        ON CREATE SET
            e.headline = $headline,
            e.score = $score
        MERGE (c)-[:HAD_EVENT]->(e)
        """
        try:
            from datetime import datetime
            ts = timestamp if timestamp else datetime.now().isoformat()

            self.neo4j_graph.run(query, 
                ticker=ticker, 
                link=link, 
                timestamp=ts,
                headline=headline, 
                score=score
            )
        except Exception as e:
            logger.error(f"Failed to create Event node for {ticker} in Neo4j: {e}")

    def add_event(self, ticker: str, headline: str, score: float, link: str):
        """
        Adds a newly detected significant event to the SQLite database
        AND adds a corresponding node to the Neo4j graph.
        """
        from datetime import datetime
        event_timestamp = datetime.now()
        timestamp_str = event_timestamp.isoformat()

        # ### CLOUD EDIT: DISABLE SQLITE WRITE ###
        # --- 1. Add to SQLite ---
        # if not self.sqlite_conn: return # <-- This guard clause would fire
        # with self.sqlite_conn as conn:
        #     conn.execute(
        #         "INSERT INTO significant_events (timestamp, ticker, headline, score, link) VALUES (?, ?, ?, ?, ?)",
        #         (event_timestamp, ticker, headline, score, link)
        #     )
        # logger.info(f" -> ✅ Event for {ticker} saved to SQLite.")

        # --- 2. Add to Neo4j (This part still runs and is what we need) ---
        self._add_event_node_to_graph(ticker, headline, score, link, timestamp_str)
        logger.info(f" -> ✅ Event node for {ticker} added to Neo4j.")

    # ### CLOUD EDIT: RE-ROUTE TO NEO4J ###
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Retrieves ALL significant events from the Neo4j database."""
        if not self.is_connected(): 
            logger.error("Neo4j connection not available.")
            return []
        
        logger.info("Retrieving all historical events from Neo4j...")
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        RETURN 
            c.ticker AS ticker, 
            e.headline AS headline, 
            e.score AS score, 
            e.link AS link, 
            e.timestamp AS timestamp
        ORDER BY e.timestamp ASC
        """
        try:
            results = self.neo4j_graph.run(query).data()
            logger.info(f" -> Found {len(results)} total events in Neo4j.")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve all events from Neo4j: {e}")
            return []

    # ### CLOUD EDIT: RE-ROUTE TO NEO4J ###
    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves the most recent significant events from the Neo4j database."""
        if not self.is_connected(): return []
        
        logger.info(f"Retrieving {limit} recent events from Neo4j...")
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        RETURN 
            c.ticker AS ticker, 
            e.headline AS headline, 
            e.score AS score, 
            e.link AS link, 
            e.timestamp AS timestamp
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        try:
            results = self.neo4j_graph.run(query, limit=limit).data()
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve recent events from Neo4j: {e}")
            return []


    # ==============================================================================
    # --- UPDATED FUNCTIONS (FIXED) ---
    # ==============================================================================

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
        """
        Builds a NetworkX DiGraph from Neo4j using the py2neo driver.
        """
        G = nx.DiGraph()
        
        if not self.is_connected():
            logger.error("get_graph_from_db: No database connection. Returning empty graph.")
            return G
        
        nodes_query = "MATCH (n:Company) RETURN n"
        
        edges_query = f"""
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > {weight_threshold}
        RETURN n.ticker AS source, m.ticker AS target, r
        """
        
        try:
            nodes_result = self.neo4j_graph.run(nodes_query)
            for record in nodes_result:
                node_data = record["n"]
                ticker = node_data['ticker']
                G.add_node(ticker, **dict(node_data))
                
            edges_result = self.neo4j_graph.run(edges_query)
            for record in edges_result:
                rel_data = record["r"]
                G.add_edge(record["source"], record["target"], **dict(rel_data))
            
        except Exception as e:
            logger.error(f"Failed to get graph from Neo4j: {e}")

        return G

    def get_neighborhood_graph(self, company_ticker: str) -> nx.DiGraph:
            """
            Gets a specific company and its 1st-degree neighbors (in and out)
            using a simpler, more robust query.
            """
            G = nx.DiGraph()

            if not self.is_connected():
                logger.error("get_neighborhood_graph: No database connection.")
                return G
            
            query = """
            MATCH (c:Company {ticker: $ticker})-[r]-(neighbor:Company)
            RETURN c, r, neighbor
            """
            
            try:
                result = self.neo4j_graph.run(query, ticker=company_ticker)
                
                for record in result:
                    center_node_data = record['c']
                    rel_data = record['r']
                    neighbor_node_data = record['neighbor']
                    
                    G.add_node(center_node_data['ticker'], **dict(center_node_data))
                    G.add_node(neighbor_node_data['ticker'], **dict(neighbor_node_data))
                    
                    G.add_edge(
                        rel_data.start_node['ticker'], 
                        rel_data.end_node['ticker'], 
                        **dict(rel_data)
                    )
                
                if G.number_of_nodes() == 0:
                    node_data_list = self.neo4j_graph.run(
                        "MATCH (c:Company {ticker: $ticker}) RETURN c", ticker=company_ticker
                    ).data()
                    if node_data_list:
                        node_data = node_data_list[0]['c']
                        G.add_node(node_data['ticker'], **dict(node_data))

            except Exception as e:
                logger.error(f"Failed to get neighborhood graph for {company_ticker}: {e}")

            return G

    # ==============================================================================
    # --- OTHER FUNCTIONS ---
    # =================================H==============================================

    def clear_neo4j_database(self):
        """
        !! DANGEROUS !! Deletes all nodes and relationships from the Neo4j database.
        """
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL DATA from the Neo4j database...")
        self.neo4j_graph.run("MATCH (n) DETACH DELETE n")
        logger.warning(" -> ✅ Neo4j database has been cleared.")

    def close(self):
        """Closes the SQLite database connection."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("SQLite connection closed.")
        # py2neo Graph object doesn't have an explicit .close()