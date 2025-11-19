# ==============================================================================
# --- IMPORTS for Background Worker ---
# ==============================================================================
import json
import time
import re
import os
import glob
import logging
import sys
from typing import Dict, Any, List

import feedparser
import spacy
import torch
from bs4 import BeautifulSoup
from polygon import RESTClient
from sec_edgar_downloader import Downloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Email Alerting Imports ---
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- Local Imports ---
from database_manager import DatabaseManager

# ==============================================================================
# --- WORKER SETUP & CONFIGURATION ---
# ==============================================================================

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FinancialWorker")

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Loads all configurations from a JSON file for the worker."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.critical(f"FATAL: Could not load config file '{config_file}'. Error: {e}")
        sys.exit(1) # Exit if config is missing, as the worker cannot run.

# ==============================================================================
# --- GLOBAL NLP MODEL LOADING ---
# ==============================================================================

@torch.no_grad()
def get_financial_sentiment(text: str, tokenizer: Any, model: Any) -> float:
    """Analyzes text using a pre-loaded FinBERT model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT: 0=positive, 1=negative, 2=neutral
        return (scores[0][0] - scores[0][1]).item()
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return 0.0

# Load models once at startup
try:
    logger.info("Loading NLP models into memory...")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp_spacy = spacy.load("en_core_web_sm")
    logger.info("✅ NLP models loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not load essential NLP models. Error: {e}")
    sys.exit(1)

# ==============================================================================
# --- ALERTING FUNCTION ---
# ==============================================================================

def generate_summary_email_body(events: list) -> str:
    """
    Generates a single HTML string to summarize a list of significant events.
    """
    header = """
    <html>
    <head>
    <style>
        body { font-family: sans-serif; margin: 20px; color: #333; }
        .container { border: 1px solid #ddd; border-radius: 8px; padding: 20px; max-width: 800px; margin: auto; }
        h2 { color: #2c3e50; }
        .event { border-top: 1px solid #eee; padding-top: 15px; margin-top: 15px; }
        .event p { margin: 5px 0; }
        .event-header { font-size: 1.1em; font-weight: bold; }
        .positive { color: #27ae60; }
        .negative { color: #c0392b; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        small { color: #7f8c8d; }
    </style>
    </head>
    <body>
    <div class="container">
    """
    
    footer = """
    <hr>
    <p><small>This is an automated summary from the Financial Causal Inference Engine.</small></p>
    </div>
    </body>
    </html>
    """

    event_html_parts = []
    for event in events:
        score_color_class = "positive" if event['score'] > 0 else "negative"
        event_html_parts.append(f"""
        <div class="event">
            <p class="event-header">
                <span class="{score_color_class}">{event['ticker']}</span>: {event['headline']}
            </p>
            <p>
                <strong>Sentiment Score:</strong> 
                <strong class="{score_color_class}">{event['score']:.2f}</strong>
            </p>
            <p><a href="{event['link']}">Read Full Article</a></p>
        </div>
        """)

    body = f"<h2>{len(events)} Significant Events Detected</h2><p>The following events met the sentiment threshold during this cycle.</p>"
    
    return header + body + "".join(event_html_parts) + footer

def send_email_alert(config: Dict[str, Any], subject: str, body_html: str):
    """Sends a formatted HTML email alert using the STARTTLS method."""
    email_keys = ['email_sender', 'email_password', 'smtp_server', 'smtp_port', 'recipient_emails']
    if not all(k in config for k in email_keys):
        logger.warning("Email configuration is incomplete. Skipping alert.")
        return

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config['email_sender']
        msg['To'] = ", ".join(config['recipient_emails'])
        msg.attach(MIMEText(body_html, 'html'))
        
        logger.info("Connecting to SMTP server to send email alert...")
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()  
            server.login(config['email_sender'], config['email_password'])
            server.send_message(msg)
            
        logger.info("✅ Email alert sent successfully.")
    except Exception as e:
        logger.error(f"Could not send email alert. Reason: {e}")

# ==============================================================================
# --- DATA PIPELINE FUNCTIONS (worker.py) ---
# ==============================================================================

def check_live_news_for_events(db_manager: DatabaseManager, config: Dict[str, Any]):
    """
    Fetches live news, collects significant events, and saves them in a batch.
    """
    logger.info("📰 Checking live news for significant events...")
    try:
        with open('news_urls.json', 'r', encoding='utf-8') as f:
            feed_urls = [s['url'] for s in json.load(f).get("sources", [])]
    except Exception as e:
        logger.error(f"Could not load news_urls.json: {e}")
        return

    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url, agent='Mozilla/5.0')
            if feed.entries: articles.extend(feed.entries)
        except Exception as e:
            logger.warning(f"Could not fetch or parse feed from {url}. Reason: {e}")

    if not articles:
        logger.info("-> No new articles found in any feeds.")
        return

    graph = db_manager.get_graph_from_db()
    all_tickers = list(graph.nodes())
    sentiment_threshold = config.get("news_sentiment_threshold", 0.7)
    # Get a set of recent headlines to skip duplicates (Good practice)
    processed_headlines = set(event['headline'] for event in db_manager.get_recent_events(limit=500))
    ticker_stopwords = {'A', 'ON', 'IT', 'HAS', 'SO', 'D', 'BE', 'ARE', 'SEE'} 

    significant_events_found = []
    for article in articles:
        title = article.get('title', 'No Title')
        if title in processed_headlines: continue

        link = article.get('link', '#')
        for ticker in all_tickers:
            if ticker in ticker_stopwords: continue

            if re.search(r'\b' + re.escape(ticker) + r'\b', title, re.IGNORECASE):
                score = get_financial_sentiment(title, finbert_tokenizer, finbert_model)
                if abs(score) >= sentiment_threshold:
                    logger.warning(f"🚨 Significant event FOUND for {ticker}! Score: {score:.2f}, Headline: {title}")
                    
                    processed_headlines.add(title)
                    
                    # --- Collect event data for the batch write later ---
                    significant_events_found.append({
                        'ticker': ticker, 
                        'headline': title,
                        'score': score, 
                        'link': link
                    })
                    # ---------------------------------------------------
                    break 

    if significant_events_found:
        logger.info(f"✍️ DB-WRITE: Writing batch of {len(significant_events_found)} new events...")
        try:
            # --- Perform the single, robust batch write ---
            db_manager.add_events_batch(significant_events_found)
            logger.info("✅ DB-WRITE: Events written to Neo4j successfully.")

            # Send email only after the database write is confirmed
            subject = f"Financial KG Summary: {len(significant_events_found)} Significant Events Detected"
            body_html = generate_summary_email_body(significant_events_found)
            send_email_alert(config, subject, body_html)
            
        except Exception as e:
            # Log a CRITICAL error if the database write failed
            logger.critical(f"FATAL DB WRITE ERROR: Could not write event batch to Neo4j. Reason: {e}", exc_info=True)
            
    else:
        logger.info("-> No new significant events found to write.")
        
    logger.info("✅ Live news check complete.")


def update_nodes_from_api(tickers: List[str], client: RESTClient, db_manager: DatabaseManager):
    """Fetches company data from the API and upserts it to the database in a batch."""
    logger.info("🔄 Updating company nodes from Polygon API...")
    nodes_to_update = []
    for ticker in tickers:
        try:
            resp = client.get_ticker_details(ticker)
            nodes_to_update.append({
                "ticker": ticker,
                "name": getattr(resp, 'name', 'N/A'),
                "sector": getattr(resp, 'sic_description', 'N/A'),
                "market_cap": getattr(resp, 'market_cap', 0)
            })
            logger.info(f"  -> Fetched data for {ticker}")
        except Exception as e:
            logger.error(f"  -> Could not fetch data for {ticker}. Reason: {e}")
        time.sleep(13)
    
    if nodes_to_update:
        db_manager.upsert_company_nodes_batch(nodes_to_update)
    logger.info("✅ Company nodes update complete.")


def add_manual_relationships(db_manager: DatabaseManager):
    """Adds a predefined set of high-conviction relationships to the database."""
    logger.info("🔗 Adding/updating manual relationships in the database...")
    edges = [
        ("KO", "KR", {"type": "dependency", "weight": 0.9, "data_source": "manual"}),
        ("OXY", "KO", {"type": "dependency", "weight": 0.6, "data_source": "manual"}),
        ("INTC", "AMD", {"type": "competitor", "weight": 0.8, "data_source": "manual"}),
        ("AMD", "INTC", {"type": "competitor", "weight": 0.8, "data_source": "manual"}),
        ("OXY", "INTC", {"type": "dependency", "weight": 0.7, "data_source": "manual"})
    ]
    for u, v, attrs in edges:
        db_manager.upsert_relationship(u, v, attrs.get('type', 'RELATED'), attrs)
    logger.info(f"  -> ✅ Wrote {len(edges)} manual relationships to the database.")

# ==============================================================================
# --- THIS IS THE UPDATED SECTION ---
# ==============================================================================
def discover_relations_from_filings(tickers: List[str], db_manager: DatabaseManager, dl: Downloader):
    """Downloads and parses SEC filings to discover new relationships."""
    logger.info("🧠 Discovering relationships from SEC filings...")
    
    logger.info("  -> Downloading recent 10-K filings...")
    for ticker in tickers:
        try:
            dl.get("10-K", ticker, limit=1, download_details=False)
        except Exception as e:
            logger.error(f"    -> Failed to download 10-K for {ticker}: {e}")

    try:
        with open('sp500_map.json', 'r', encoding='utf-8') as f:
            company_map = {item['ticker']: item['name'] for item in json.load(f) if 'name' in item}
    except FileNotFoundError:
        logger.error("  -> 'sp500_map.json' not found. Cannot perform entity matching.")
        return

    # --- BATCHING LOGIC START ---
    # Create containers to hold all discovered items *before* writing to the DB
    all_new_relationships = []
    all_new_nodes_map = {}       # Use a dict/map to store unique nodes, keyed by ticker
    collected_pairs_global = set() # Use a set to track all unique pairs found (A, B)
    # --- BATCHING LOGIC END ---

    for ticker in tickers:
        search_path = os.path.join('sec-edgar-filings', ticker, '10-K', '*', 'full-submission.txt')
        filing_paths = glob.glob(search_path)
        if not filing_paths: continue

        logger.info(f"--- Processing filing for {ticker} ---")
        try:
            with open(filing_paths[0], 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                full_text = soup.get_text(separator=' ', strip=True)
            
            if not full_text or len(full_text) < 1000: continue
            
            doc = nlp_spacy(full_text[:nlp_spacy.max_length])

            for sent in doc.sents:
                org_entities = [ent.text.lower() for ent in sent.ents if ent.label_ == 'ORG']
                if not org_entities: continue

                for related_ticker, name in company_map.items():
                    if ticker == related_ticker: continue
                    
                    # Create a sorted pair to uniquely identify this relationship
                    pair = tuple(sorted((ticker, related_ticker)))
                    
                    # If we have *already* found this pair in a previous filing, skip
                    if pair in collected_pairs_global: continue

                    name_lower = name.lower()
                    if any(org in name_lower or name_lower in org for org in org_entities):
                        sentiment = get_financial_sentiment(sent.text, finbert_tokenizer, finbert_model)
                        weight = min(1.0, 0.5 + (0.2 * sentiment))
                        attrs = {"type": "sec_discovered", "weight": round(weight, 2), "sentiment": round(sentiment, 2)}
                        
                        # --- BATCHING LOGIC (COLLECTION) ---
                        # 1. Add the new node to our map (dict). This automatically handles duplicates.
                        if related_ticker not in all_new_nodes_map:
                            all_new_nodes_map[related_ticker] = {
                                "ticker": related_ticker, "name": name, "sector": "Discovered", "market_cap": 0
                            }

                        # 2. Add the relationship tuple to our list
                        all_new_relationships.append((ticker, related_ticker, "SEC_DISCOVERED", attrs))
                        
                        logger.info(f"  -> ✅ DISCOVERED: Found relationship {ticker} -> {related_ticker} (Weight: {weight:.2f})")
                        
                        # 3. Mark this pair as "collected" so we don't add it again
                        collected_pairs_global.add(pair)
                        
        except Exception as e:
            logger.error(f"  -> Failed to process filing for {ticker}. Reason: {e}")

    # --- BATCHING LOGIC (WRITE) ---
    # Now, *after* all filings are processed, write all collected items to the database.
    
    # 1. Write all unique new nodes in a single batch
    if all_new_nodes_map:
        nodes_list = list(all_new_nodes_map.values())
        logger.info(f"  -> ✍️ DB-WRITE: Upserting batch of {len(nodes_list)} newly discovered company nodes...")
        db_manager.upsert_company_nodes_batch(nodes_list)
    else:
        logger.info("  -> No new company nodes were discovered.")

    # 2. Write all unique new relationships
    if all_new_relationships:
        logger.info(f"  -> ✍️ DB-WRITE: Upserting {len(all_new_relationships)} newly discovered unique relationships...")
        for u, v, rel_type, attrs in all_new_relationships:
            db_manager.upsert_relationship(u, v, rel_type, attrs)
        logger.info(f"  -> ✅ DB-WRITE: Wrote {len(all_new_relationships)} unique new relationships.")
    else:
        logger.info("  -> No new relationships were discovered.")

    logger.info("✅ SEC relationship discovery complete.")

# ==============================================================================
# --- END OF UPDATED SECTION ---
# ==============================================================================


# ==============================================================================
# --- MAIN WORKER ORCHESTRATION ---
# ==============================================================================

def run_full_data_pipeline():
    """The main function for the worker, orchestrating all data tasks."""
    config = load_config()
    db_manager = DatabaseManager(config)
    if not db_manager.is_connected():
        logger.critical("Cannot run pipeline without a valid database connection.")
        return

    client = RESTClient(config.get("polygon_api_key"))
    dl = Downloader(config.get("downloader_company"), config.get("downloader_email"))
    tickers = config.get("target_tickers", [])
    
    logger.info("\n" + "="*60 + f"\n🚀 STARTING DATA REFRESH PIPELINE\n" + "="*60)
    try:
        update_nodes_from_api(tickers, client, db_manager)
        add_manual_relationships(db_manager)
        discover_relations_from_filings(tickers, db_manager, dl)
        check_live_news_for_events(db_manager, config)
    except Exception as e:
        logger.critical(f"A critical error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        db_manager.close() # Close SQLite connection
    
    logger.info("\n" + "="*60 + f"\n✅ PIPELINE COMPLETE\n" + "="*60 + "\n")


if __name__ == "__main__":
    # This script is designed to be run once by a scheduler (like cron).
    run_full_data_pipeline()