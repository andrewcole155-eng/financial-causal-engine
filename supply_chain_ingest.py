# ==============================================================================
# --- SUPPLY CHAIN MINER (Local 10-K Version) ---
# ==============================================================================
# Scans local SEC filings to extract supplier/customer relationships using Gemini.
# ==============================================================================

import os
import json
import logging
import glob
import time
import re
from google import genai
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SupplyChain")

# --- CONFIGURATION ---
FILINGS_DIR = "/home/andrew/.ssh/Trading/Knowledge_Graph/sec-edgar-filings"

def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

def clean_sec_text(text):
    """
    Simple cleaner to remove the massive XML/XBRL headers often found
    at the start of full-submission.txt files, so we get to the real text faster.
    """
    # Find the first occurrence of "Item 1" or "ITEM 1" which usually starts the Business section
    # This skips the Table of Contents and Headers
    match = re.search(r'(item\s+1\.\s+business)', text, re.IGNORECASE)
    if match:
        return text[match.start():]
    
    # If not found, just return the whole thing but stripping generic tags might help
    return text

def extract_relationships_from_text(ticker, text, client):
    """
    Uses Gemini to find supply chain links in the 10-K text.
    """
    # Clean the text to skip headers
    relevant_text = clean_sec_text(text)
    
    # Use a MUCH larger context window (500k chars is ~100k tokens, well within Flash's 1M limit)
    # This ensures we catch Item 1 (Business) and Item 1A (Risk Factors)
    excerpt = relevant_text[:500000] 
    
    prompt = f"""
    Analyze this 10-K filing text for {ticker}.
    You are a financial analyst looking for specific company names mentioned as:
    1. Suppliers / Vendors / Manufacturers
    2. Major Customers / Clients (>=10% revenue)
    3. Competitors
    4. Strategic Partners
    
    Output strictly valid JSON in this format:
    {{
        "relationships": [
            {{"target_ticker": "TSM", "target_name": "TSMC", "type": "SUPPLIES_TO"}},
            {{"target_ticker": "AAPL", "target_name": "Apple", "type": "BUYS_FROM"}},
            {{"target_ticker": "INTC", "target_name": "Intel", "type": "COMPETES_WITH"}}
        ]
    }}
    
    Rules:
    - Ignore generic terms like "governments", "distributors", "original equipment manufacturers".
    - Only extract specific capitalized Company Names.
    - Guess the TICKER if obvious (e.g., "Microsoft" -> "MSFT"). If unknown, leave null.
    - If the text contains XML/HTML formatting, ignore the tags and focus on the content.
    
    Text Excerpt:
    "{excerpt}"
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"   ❌ AI Analysis failed for {ticker}: {e}")
        return {"relationships": []}

def ingest_supply_chain():
    config = load_config()
    if not config: return

    # 1. Setup Gemini
    api_key = config.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("❌ GOOGLE_API_KEY not found.")
        return
        
    client = genai.Client(api_key=api_key)
    db = DatabaseManager(config)
    
    # 2. Find Ticker Folders
    if not os.path.exists(FILINGS_DIR):
        logger.error(f"❌ Filings directory not found: {FILINGS_DIR}")
        return

    ticker_folders = [f for f in os.listdir(FILINGS_DIR) if os.path.isdir(os.path.join(FILINGS_DIR, f))]
    logger.info(f"📂 Found {len(ticker_folders)} ticker folders to scan.")
    
    for ticker in ticker_folders:
        ticker = ticker.upper()
        
        # --- PATH LOGIC ---
        # Look recursively for txt files
        ticker_base_path = os.path.join(FILINGS_DIR, ticker)
        
        try:
            txt_files = glob.glob(f"{ticker_base_path}/**/*.txt", recursive=True)
            
            if not txt_files:
                logger.warning(f"   ⚠️ No text files found for {ticker}")
                continue
                
            # Pick largest file
            target_file = max(txt_files, key=os.path.getsize)
            
            logger.info(f"📖 Reading 10-K for {ticker}...")
            
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 3. Ask AI to Extract (Now with MASSIVE context)
            data = extract_relationships_from_text(ticker, content, client)
            
            relationships = data.get("relationships", [])
            
            if not relationships:
                logger.info(f"   -> No specific links found for {ticker}.")
            
            # 4. Write to Neo4j
            count_new = 0
            for rel in relationships:
                target_ticker = rel.get("target_ticker")
                target_name = rel.get("target_name")
                rel_type = rel.get("type")
                
                if target_ticker and target_ticker != "null":
                    if target_ticker == ticker: continue

                    # Ensure Target Node
                    db.execute_write("""
                    MERGE (t:Company {ticker: $ticker})
                    ON CREATE SET t.name = $name, t.sector = 'Unknown', t.source = '10K_Mining'
                    """, ticker=target_ticker, name=target_name)
                    
                    # Create Relationship
                    query = f"""
                    MATCH (source:Company {{ticker: $source}})
                    MATCH (target:Company {{ticker: $target}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.weight = 0.9,
                        r.mechanism = '10-K Disclosure',
                        r.verification_status = 'VERIFIED_FILING'
                    """
                    db.execute_write(query, source=ticker, target=target_ticker)
                    count_new += 1
            
            if count_new > 0:
                logger.info(f"   ✅ Added {count_new} relationships for {ticker}.")
            
            # Small sleep to prevent rate limit spikes
            time.sleep(2)

        except Exception as e:
            logger.error(f"   ❌ Error processing {ticker}: {e}")

    logger.info("✅ Supply Chain Mining Complete.")
    db.close()

if __name__ == "__main__":
    ingest_supply_chain()