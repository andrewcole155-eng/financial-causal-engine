# data_processing/fetchers.py

import asyncio
import aiohttp
import feedparser
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple

# --- Import from the nlp module in the same package ---
from .nlp import master_text_preprocessor

# --- Setup structured logging ---
# This is better than print() for managing application output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def fetch_html(url: str, session: aiohttp.ClientSession) -> Tuple[Optional[str], str]:
    """Fetches HTML content asynchronously."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        async with session.get(url, timeout=30, headers=headers, ssl=False) as response:
            response.raise_for_status()
            return await response.text(errors='ignore'), url
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None, url


async def fetch_and_process_html(urls: List[str]) -> List[str]:
    """Fetches and processes a list of HTML webpages asynchronously."""
    connector = aiohttp.TCPConnector(limit=15, limit_per_host=5, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_html(url, session) for url in urls]
        html_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in html_results:
        if isinstance(result, Exception): continue
        html_content, url = result
        if html_content:
            word_list = master_text_preprocessor(html_content)
            if word_list:
                processed_results.append(" ".join(word_list).lower())
    return processed_results


def fetch_live_news_from_feeds(feed_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches and structures news articles from RSS feeds.
    Returns a list of dictionaries, each representing a news article.
    """
    logger.info("📰 Fetching live news from RSS feeds...")
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url, agent='Mozilla/5.0')
            if not feed.entries:
                continue
            for entry in feed.entries:
                articles.append({
                    'title': entry.get('title', 'No Title'),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'source': feed.feed.get('title', url)
                })
        except Exception as e:
            logger.error(f"Error processing feed {url}: {e}")
    logger.info(f"  -> Fetched {len(articles)} total articles.")
    return articles


def fetch_historical_prices(ticker: str, start_date: Any, end_date: Any, client: Any) -> Optional[pd.DataFrame]:
    """
    Fetches historical daily stock prices for a ticker and returns them as a Pandas DataFrame.
    """
    try:
        logger.info(f" -> Fetching historical prices for {ticker} from {start_date} to {end_date}...")
        # Note: 'client' is of type polygon.RESTClient, using Any for simplicity if not importing the class
        aggs = client.get_aggs(ticker, 1, "day", str(start_date), str(end_date))
        
        if not aggs:
            logger.warning(f"     -> No price data found for {ticker} in the given range.")
            return None
            
        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date')
        
        logger.info(f"     -> Successfully fetched {len(df)} days of price data for {ticker}.")
        return df
    except Exception as e:
        logger.error(f"     -> An error occurred while fetching prices for {ticker}: {e}")
        return None