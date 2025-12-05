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
GEMINI_KEY = config.get("GOOGLE_API_KEY")
POLYGON_KEY = config.get("polygon_api_key") # Ensure this is in your config.json

# --- AI SETUP ---
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    logger.warning("⚠️ No Gemini API Key found. Sentiment will be 0.0")
    model = None

# --- AI SCORING (Sync) ---
# --- UPDATED AI SCORING (With Relevance/Confidence) ---
def get_ai_analysis(headline, tickers):
    """
    Returns a tuple: (sentiment_score, relevance_score)
    """
    if not model: return 0.0, 0.0
    try:
        # We pass the tickers so the AI knows who the news SHOULD be about
        prompt = f"""
        Analyze this financial news headline for the specific tickers: {tickers}.
        Headline: "{headline}"

        Task:
        1. Calculate Sentiment Score (-1.0 Negative to 1.0 Positive).
        2. Calculate Relevance Score (0.0 to 1.0). 
           - Ask yourself: Is this news REALLY about the company {tickers}?
           - Example of LOW relevance: "Top cop retired" (Police) tagged for ticker "COP" (ConocoPhillips).
           - Example of LOW relevance: "ServiceNow brand voice" tagged for ticker "NOW".

        Return ONLY a JSON object with keys 'score' and 'relevance'.
        Example: {{"score": -0.5, "relevance": 0.1}}
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        return float(data.get("score", 0.0)), float(data.get("relevance", 0.0))
        
    except Exception as e:
        logger.error(f"AI Scoring Failed: {e}")
        return 0.0, 0.0

# --- MAIN LOOP ---
def main():
    if not POLYGON_KEY:
        logger.error("❌ Polygon API Key not found in config.json")
        return

    # Initialize Managers
    db = DatabaseManager(config)
    client = RESTClient(api_key=POLYGON_KEY)
    
    # Track seen articles to prevent duplicates
    seen_news_ids = set()
    
    # We increase the default poll to 35s to be safe on Free Tier
    POLL_INTERVAL = 35 
    
    logger.info(f"✅ Pulse Active (Polygon Mode). Polling every {POLL_INTERVAL}s.")
    logger.info("ℹ️  Confidence Filter Active: Events with < 50% relevance will be discarded.")

    while True:
        try:
            # 1. Fetch the generator (Limit 10 to catch bursts)
            # We wrap this specifically to catch connection/rate limit errors early
            try:
                news_generator = client.list_ticker_news(limit=10)
                
                # 2. Manually take only the first 10 items
                latest_batch = []
                for _ in range(10):
                    try:
                        latest_batch.append(next(news_generator))
                    except StopIteration:
                        break # End of available news
            
            except Exception as api_error:
                # Check for Rate Limit (429) specifically
                if "429" in str(api_error) or "Max retries" in str(api_error):
                    logger.warning("🛑 Hit Polygon Rate Limit (429). Sleeping 60s to reset...")
                    time.sleep(60)
                    continue # Skip the rest of this loop and try again
                else:
                    raise api_error # Re-raise other errors to be caught below

            # 3. Reverse to process oldest -> newest
            articles = reversed(latest_batch)

            for item in articles:
                # Deduplication check
                if item.id in seen_news_ids:
                    continue
                
                seen_news_ids.add(item.id)
                if len(seen_news_ids) > 1000: # Keep memory clean
                    seen_news_ids.pop()

                headline = item.title
                tickers = item.tickers if item.tickers else []
                url = item.article_url

                if not tickers or not headline: 
                    continue

                # 4. Get Score AND Relevance (Metacognition)
                # Note: Ensure get_ai_analysis catches its own errors internally if possible
                score, relevance = get_ai_analysis(headline, tickers)

                # 5. The Confidence Filter
                if relevance < 0.5:
                    logger.info(f"🗑️ REJECTED (Noise): {headline[:30]}... (Rel: {relevance:.2f})")
                    continue

                # 6. Filter Noise (Only log significant sentiment)
                if abs(score) >= 0.15:
                    logger.info(f"⚡ NEWS (Sc: {score} | Rel: {relevance}): {headline[:50]}... {tickers}")
                    
                    # Update Database and Graph
                    for ticker in tickers:
                        # --- FIX: Sanitize Ticker (BRK.B -> BRK-B) ---
                        safe_ticker = ticker.replace('.', '-') 
                        
                        # Insert into SQLite
                        db.insert_event(
                            ticker=safe_ticker,  # Use safe_ticker
                            headline=headline,
                            score=score,
                            link=url
                        )

                        # Update Neo4j Graph
                        query = """
                        MATCH (c:Company {ticker: $ticker})
                        ...
                        """
                        # Pass safe_ticker to the query
                        db.execute_write(query, ticker=safe_ticker, score=score, relevance=relevance, headline=headline, url=url)
            
            # 7. Standard Rate Limit Wait
            # Using 35s ensures we never accidentally hit the 5 calls/min limit
            time.sleep(POLL_INTERVAL)

        except (RequestException, MaxRetryError, NewConnectionError) as e:
            logger.error(f"❌ Connection/API Error: {e}")
            logger.info("💤 Network unstable. Sleeping 60s...")
            time.sleep(60)

        except Exception as e:
            logger.error(f"❌ Unexpected Polling Error: {e}")
            time.sleep(35) # Wait before retrying

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pulse stopped by user.")