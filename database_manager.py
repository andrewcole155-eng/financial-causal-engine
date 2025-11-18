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

        # ### UNIVERSAL EDIT: Re-enable SQLite but make it fail-safe ###
        try:
            db_dir = 'database'
            db_path = os.path.join(db_dir, 'financial_data.db')
            os.makedirs(db_dir, exist_ok=True) 
            self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._create_sqlite_tables()
            logger.info(f" -> ✅ Successfully connected to SQLite at {db_path}")
        except Exception as e:
            logger.warning(f" -> ⚠️ Failed to connect to local SQLite (this is normal on Streamlit): {e}")
            self.sqlite_conn = None

        # --- Connect to Neo4j ---
        try:
            # ### UNIVERSAL EDIT: Handle BOTH config formats ###
            neo4j_uri = None
            neo4j_user = None
            neo4j_password = None

            if "neo4j" in config and isinstance(config.get("neo4j"), dict):
                logger.info(" -> Reading Neo4j config from nested [neo4j] block (Streamlit mode).")
                neo4j_uri = config["neo4j"].get("uri")
                neo4j_user = config["neo4j"].get("user")
                neo4j_password = config["neo4j"].get("password")
            
            elif "neo4j_uri" in config:
                logger.info(" -> Reading Neo4j config from flat 'neo4j_uri' keys (Worker/Backfill mode).")
                neo4j_uri = config.get("neo4j_uri")
                neo4j_user = config.get("neo4j_user")
                neo4j_password = config.get("neo4j_password")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection details not found in any config source.")

            self.neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_graph.run("MATCH (n) RETURN count(n)")
            logger.info(" -> ✅ Successfully connected to Neo4j.")
            
        except Exception as e:
            logger.critical(f" -> ❌ FATAL: Failed to connect to Neo4j: {e}")
            self.neo4j_graph = None
#s
    def is_connected(self) -> bool:
        """Checks if the connection to Neo4j is active."""
        return self.neo4j_graph is not None

    def _create_sqlite_tables(self):
        """Creates all necessary tables in the SQLite database if they don't exist."""
        if not self.sqlite_conn: 
            logger.info(" -> Skipping SQLite table creation (no connection).")
            return
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
        This now also updates the timestamp ON MATCH.
        """
        if not self.is_connected(): return
        
        # This query ensures the Company node exists. If not, it fails gracefully.
        query = """
        MATCH (c:Company {ticker: $ticker})
        MERGE (e:Event {link: $link})
        ON CREATE SET
            e.headline = $headline,
            e.score = $score,
            e.timestamp = $timestamp
        ON MATCH SET
            e.headline = $headline,
            e.score = $score,
            e.timestamp = $timestamp
        MERGE (c)-[:HAD_EVENT]->(e)
        """
        try:
            from datetime import datetime
            # Ensure timestamp is valid, default to now if not
            ts = timestamp if timestamp else datetime.now().isoformat()
            
            self.neo4j_graph.run(query, 
                ticker=ticker, 
                link=link, 
                timestamp=ts,
                headline=headline, 
                score=score
            )
        except Exception as e:
            # This will log if the MATCH (c:Company) fails
            logger.error(f"Failed to create Event node for {ticker} in Neo4j: {e}")

    def add_event(self, ticker: str, headline: str, score: float, link: str):
        """
        Adds a newly detected significant event to the SQLite database (if available)
        AND adds a corresponding node to the Neo4j graph.
        """
        from datetime import datetime
        event_timestamp = datetime.now()
        timestamp_str = event_timestamp.isoformat()

        if self.sqlite_conn:
            try:
                with self.sqlite_conn as conn:
                    conn.execute(
                        "INSERT INTO significant_events (timestamp, ticker, headline, score, link) VALUES (?, ?, ?, ?, ?)",
                        (event_timestamp, ticker, headline, score, link)
                    )
                logger.info(f" -> ✅ Event for {ticker} saved to SQLite.")
            except Exception as e:
                logger.error(f" -> Failed to save event for {ticker} to SQLite: {e}")
        else:
            logger.info(" -> Skipping SQLite event logging (no connection).")

        self._add_event_node_to_graph(ticker, headline, score, link, timestamp_str)
        logger.info(f" -> ✅ Event node for {ticker} added to Neo4j.")

    def get_all_events(self) -> List[Dict[str, Any]]:
        """
        Retrieves ALL significant events from the Neo4j database.
        This is for the Streamlit app.
        """
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

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent significant events from the Neo4j database.
        This is for the Streamlit app.
        """
        if not self.is_connected(): return []
        
        logger.info(f"Retrieving {limit} recent events from Neo4j...")
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        WHERE e.timestamp IS NOT NULL
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

    # ### NEW FUNCTION FOR BACKFILL SCRIPT ###
    def get_all_events_from_sqlite(self) -> List[Dict[str, Any]]:
        """
        Retrieves ALL significant events from the LOCAL SQLITE database.
        This is ONLY for the backfill script.
        """
        if not self.sqlite_conn: 
            logger.error("SQLite connection not available. Cannot backfill.")
            return []
        
        logger.info("Retrieving all historical events from SQLite for backfill...")
        try:
            with self.sqlite_conn as conn:
                # Order by ID to get them in the order they were found
                cursor = conn.execute("SELECT * FROM significant_events ORDER BY id ASC")
                events = [dict(row) for row in cursor.fetchall()]
                logger.info(f" -> Found {len(events)} total events in SQLite.")
                return events
        except Exception as e:
            logger.error(f"Failed to retrieve all events from SQLite: {e}")
            return []

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
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
            G = nx.DiGraph()
            if not self.is_connected():
                logger.error("get_neighborhood_graph: No database connection.")
                return G
            
            # This query is correct: It gets the Top 25 relationships
            query = """
            MATCH (c:Company {ticker: $ticker})-[r]-(neighbor:Company)
            RETURN c, r, neighbor
            ORDER BY r.weight DESC
            LIMIT 25
            """
            
            try:
                result = self.neo4j_graph.run(query, ticker=company_ticker)
                
                nodes_added = set()
                
                for record in result:
                    center_node_data = record['c']
                    rel_data = record['r']
                    neighbor_node_data = record['neighbor']
                    
                    center_ticker = center_node_data['ticker']
                    neighbor_ticker = neighbor_node_data['ticker']

                    # Add the center node if we haven't already
                    if center_ticker not in nodes_added:
                        G.add_node(center_ticker, **dict(center_node_data))
                        nodes_added.add(center_ticker)
                        
                    # Add the neighbor node if we haven't already
                    if neighbor_ticker not in nodes_added:
                        G.add_node(neighbor_ticker, **dict(neighbor_node_data))
                        nodes_added.add(neighbor_ticker)
                    
                    # --- THIS IS THE FIX ---
                    # I have removed the extra 'type=' argument which was
                    # conflicting with the 'type' property inside dict(rel_data).
                    # This is the original, correct code.
                    G.add_edge(
                        rel_data.start_node['ticker'], 
                        rel_data.end_node['ticker'], 
                        **dict(rel_data)
                    )
                    # --- END FIX ---
                
                # This correctly handles companies with 0 relationships
                if G.number_of_nodes() == 0:
                    node_data_list = self.neo4j_graph.run(
                        "MATCH (c:Company {ticker: $ticker}) RETURN c", ticker=company_ticker
                    ).data()
                    if node_data_list:
                        node_data = node_data_list[0]['c']
                        G.add_node(node_data['ticker'], **dict(node_data))

            except Exception as e:
                # This log will catch any future errors
                logger.error(f"Failed to build neighborhood graph for {company_ticker}: {e}")

            return G

    def clear_neo4j_database(self):
        """
        !! DANGEROUS !! Deletes all nodes and relationships from the Neo4j database.
        """
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL DATA from the Neo4j database...")
        self.neo4j_graph.run("MATCH (n) DETACH DELETE n")
        logger.warning(" -> ✅ Neo4j database has been cleared.")

    def clear_neo4j_events(self):
        """
        Deletes all :Event nodes and their :HAD_EVENT relationships from Neo4j.
        """
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL :Event nodes and :HAD_EVENT relationships from Neo4j...")
        self.neo4j_graph.run("MATCH (e:Event) DETACH DELETE e")
        logger.warning(" -> ✅ Neo4j events have been cleared.")


    def close(self):
        """Closes the SQLite database connection."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info("SQLite connection closed.")
        # py2neo Graph object doesn't have an explicit .close()