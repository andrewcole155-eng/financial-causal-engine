import google.generativeai as genai
import json
import os
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_api_key():
    """Loads Google API Key from config.json or Environment."""
    try:
        # Priority 1: Check Environment
        if os.getenv("GOOGLE_API_KEY"):
            return os.getenv("GOOGLE_API_KEY")
            
        # Priority 2: Check Config File
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('GOOGLE_API_KEY')
    except Exception:
        return None

def generate_financial_narrative(ticker, prediction_label, triples):
    """
    Sends serialized graph triples to Gemini to generate a text narrative.
    """
    api_key = load_api_key()
    if not api_key:
        return "⚠️ Error: GOOGLE_API_KEY not found in config.json or environment."

    genai.configure(api_key=api_key)

    if not triples:
        return "No sufficient causal paths found to generate a narrative."

    # 1. Format the Causal Facts
    context_str = "\n".join(triples)
    
    # 2. Strict Analyst Prompt
    prompt = f"""
    ROLE: You are a Senior Quantitative Analyst at a Hedge Fund.
    TASK: Explain the results of a Causal Graph Neural Network (GNN) to a Portfolio Manager.
    
    CONTEXT: 
    The GNN analyzed **{ticker}** and predicted: **{prediction_label}**.
    
    EVIDENCE (Strictly based on GNNExplainer output):
    ---------------------------------------------------------
    {context_str}
    ---------------------------------------------------------
    
    INSTRUCTIONS:
    1. Write a concise, 3-4 sentence paragraph explaining the prediction.
    2. You MUST reference the specific relationships listed in the EVIDENCE (e.g., "driven by rising Oil prices" or "linked to Supplier X").
    3. Do NOT mention "nodes", "edges", or "weights". Use natural financial language.
    4. Do NOT hallucinate external news. Only use the provided evidence.
    """

    try:
        # Use Flash for speed/cost efficiency
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini Narrative Failed: {e}")
        return f"Error generating narrative: {str(e)}"