# data_processing/nlp.py

import nltk
import html
import re
import logging
from typing import List, Set, Any

from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from stop_words import get_stop_words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_download_nltk_data():
    """Checks for necessary NLTK data and downloads if missing."""
    required_data = [
        'corpora/stopwords', 'tokenizers/punkt', 'taggers/averaged_perceptron_tagger',
        'chunkers/maxent_ne_chunker', 'corpora/words', 'sentiment/vader_lexicon'
    ]
    try:
        for data in required_data:
            nltk.data.find(data)
        logger.info("All necessary NLTK data found.")
    except LookupError:
        logger.warning("Downloading necessary NLTK data for NLP module...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

# Run the check once when the module is imported
check_and_download_nltk_data()

_stopwords_cache: Set[str] = set()
def get_stopwords() -> Set[str]:
    """Builds and returns a memoized set of stopwords."""
    global _stopwords_cache
    if _stopwords_cache:
        return _stopwords_cache
    
    stop_words = set(get_stop_words('en'))
    nltk_stops = set(nltk.corpus.stopwords.words("english"))
    _stopwords_cache = stop_words.union(nltk_stops)
    return _stopwords_cache

stopwords_set = get_stopwords()

def master_text_preprocessor(text_content: str) -> List[str]:
    """A robust function to clean raw text from any source."""
    if not text_content or not isinstance(text_content, str):
        return []
    try:
        text = html.unescape(text_content)
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        # Remove URLs, non-alphabetic characters, and extra whitespace
        clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Tokenize, preprocess, and remove stopwords
        words = simple_preprocess(clean_text, deacc=True, min_len=3)
        return [word for word in words if word not in stopwords_set]
    except Exception as e:
        logger.error(f"Error in master_text_preprocessor: {e}")
        return []

def get_headline_sentiment(headline_text: str) -> float:
    """Analyzes a headline using VADER and returns its compound sentiment score."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(headline_text)
    return score['compound']

def get_financial_sentiment(headline_text: str, tokenizer: Any, model: Any) -> float:
    """
    Analyzes a headline using a pre-trained financial sentiment model (like FinBERT).
    Returns a single score from -1 (very negative) to +1 (very positive).
    """
    try:
        # Note: 'tokenizer' and 'model' are from the transformers library, using Any for simplicity
        inputs = tokenizer(headline_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT labels: 0 -> positive, 1 -> negative, 2 -> neutral
        # We calculate a simple positive - negative score
        positive_score = scores[0][0].item()
        negative_score = scores[0][1].item()
        
        return positive_score - negative_score
    except Exception as e:
        logger.error(f"Error during financial sentiment analysis for text: '{headline_text}'. Error: {e}")
        return 0.0