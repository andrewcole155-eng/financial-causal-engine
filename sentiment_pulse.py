# ==============================================================================
# --- SENTIMENT PULSE (Polygon.io Edition) ---
# ==============================================================================
# Polls Polygon.io News (Safe for Free Tier Limits), updates risk scores via 
# Gemini AI, and logs Events to the Graph.
#
# NOTE: This script runs 24/7 to catch breaking news outside market hours.
# During Market Hours (9:30-4:00 ET), the Alpaca Trading Bot should also be running.
# ==============================================================================

import json
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta
import google.generativeai as genai
from polygon import RESTClient
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Pulse")

# --- CONFIGURATION ---
def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

config = load_config()
GEMINI_KEY = config.get("GEMINI_API_KEY")
POLYGON_KEY = config.get("polygon_api_key") # Ensure this is in your config.json

# --- AI SETUP ---
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    logger.warning("⚠️ No Gemini API Key found. Sentiment will be 0.0")
    model = None

# --- AI SCORING (Sync) ---
def get_ai_score_sync(headline):
    if not model: return 0.0
    try:
        prompt = f"""
        Analyze the sentiment of this financial news headline.
        Headline: "{headline}"
        Return ONLY a JSON object with a single key 'score' (float -1.0 to 1.0).
        """
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return float(json.loads(text).get("score", 0.0))
    except Exception as e:
        logger.error(f"AI Scoring Failed: {e}")
        return 0.0

# --- MAIN LOOP ---
def main():
    if not POLYGON_KEY:
        logger.error("❌ Polygon API Key not found in config.json")
        return

    # Initialize Managers
    db = DatabaseManager(config)
    client = RESTClient(api_key=POLYGON_KEY)
    
    # Track seen articles to prevent duplicates
    # (We store the last 100 IDs in memory)
    seen_news_ids = set()
    
    logger.info("✅ Pulse Active (Polygon Mode). Polling every 15s to adhere to API limits.")
    logger.info("ℹ️  Note: This monitors news 24/7. Ensure Trading Bot runs during market hours.")

    while True:
        try:
            # 1. Poll Polygon for latest news (Limit 10 to catch bursts)
            # This counts as 1 API Call. Safe limit is 5/min.
            # We poll every 15s = 4 calls/min.
            response = client.list_ticker_news(limit=10)

            # Process from oldest to newest in this batch to maintain timeline
            articles = reversed(list(response)) 

            for item in articles:
                # Deduplication check
                if item.id in seen_news_ids:
                    continue
                
                seen_news_ids.add(item.id)
                if len(seen_news_ids) > 1000: # Keep memory clean
                    seen_news_ids.pop()

                headline = item.title
                # Polygon gives tickers as a list ["AAPL", "MSFT"]
                tickers = item.tickers if item.tickers else []
                url = item.article_url

                if not tickers or not headline: 
                    continue

                # 2. Get Sentiment Score
                score = get_ai_score_sync(headline)

                # 3. Filter Noise (Only log significant news)
                if abs(score) >= 0.15:
                    logger.info(f"⚡ NEWS ({score}): {headline[:50]}... {tickers}")
                    
                    for ticker in tickers:
                        # Insert into SQLite (for UI)
                        db.insert_event(
                            ticker=ticker,
                            headline=headline,
                            score=score,
                            link=url
                        )

                        # Update Neo4j Graph
                        query = """
                        MATCH (c:Company {ticker: $ticker})
                        SET c.sentiment_score = $score,
                            c.last_news_update = datetime()
                        
                        CREATE (e:Event {
                            headline: $headline,
                            score: $score,
                            source: 'Polygon_Gemini',
                            timestamp: datetime(),
                            link: $url
                        })
                        MERGE (c)-[:HAD_EVENT]->(e)
                        """
                        db.execute_write(query, ticker=ticker, score=score, headline=headline, url=url)
            
            # 4. Rate Limit Wait (CRITICAL)
            # Free Tier Limit: 5 req/min. 
            # Sleep 15s ensures max 4 req/min.
            time.sleep(15)

        except Exception as e:
            logger.error(f"Polling Error: {e}")
            time.sleep(15) # Wait before retrying

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pulse stopped by user.")