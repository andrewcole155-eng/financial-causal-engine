import json
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import os
import streamlit as st  # Required for Cloud Secrets

# ==============================================================================
# --- 1. ROBUST CONFIGURATION LOADER ---
# ==============================================================================
def get_fred_api_key():
    """
    Tries to load the API Key from:
    1. Streamlit Secrets (Cloud Deployment)
    2. config.json (Local Development)
    3. Environment Variables (Docker/System)
    """
    # 1. Try Streamlit Secrets (Cloud)
    if hasattr(st, "secrets") and "fred_api_key" in st.secrets:
        return st.secrets["fred_api_key"]
    
    # 2. Try Local config.json
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                return config.get('fred_api_key')
    except Exception as e:
        print(f"Warning: Could not read config.json: {e}")
        
    # 3. Try Environment Variable
    return os.environ.get("FRED_API_KEY")

# Initialize Fred API
api_key = get_fred_api_key()

if not api_key:
    print("⚠️ WARNING: FRED API Key not found. Macro data fetching will fail.")
    fred = None
else:
    try:
        fred = Fred(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Fred API: {e}")
        fred = None

# ==============================================================================
# --- 2. SAHM RULE LOGIC (Recession Detector) ---
# ==============================================================================
def calculate_sahm_rule():
    """
    Fetches UNRATE (Unemployment Rate), calculates the 3-month moving average, 
    and compares it to the lowest 3-month average in the previous 12 months.
    """
    if not fred:
        return {"sahm_signal_value": 0.0, "is_recession_active": False}

    try:
        # Fetch Unemployment Rate (UNRATE) - Get last 2 years to be safe
        unrate = fred.get_series('UNRATE', observation_start=datetime(datetime.now().year - 2, 1, 1))
        
        # 1. Calculate 3-month moving average
        unrate_3m_avg = unrate.rolling(window=3).mean()
        
        # 2. Calculate the lowest 3-month avg in the previous 12 months
        # We shift by 1 to ensure we are looking at the *previous* 12 months, excluding current
        low_last_12m = unrate_3m_avg.rolling(window=12).min()
        
        # 3. Calculate Sahm Signal (Current 3m Avg - Lowest in last 12m)
        sahm_signal = unrate_3m_avg - low_last_12m
        
        # Get the latest available value (drop NaNs)
        valid_signal = sahm_signal.dropna()
        if valid_signal.empty:
             return {"sahm_signal_value": 0.0, "is_recession_active": False}

        current_signal = valid_signal.iloc[-1]
        
        # Boolean Switch: Sahm Rule triggers if signal >= 0.50
        is_recession = current_signal >= 0.50
        
        return {
            "sahm_signal_value": round(float(current_signal), 4),
            "is_recession_active": bool(is_recession),
            "unemployment_rate": float(unrate.iloc[-1])
        }
    except Exception as e:
        print(f"Error calculating Sahm Rule: {e}")
        return {"sahm_signal_value": 0.0, "is_recession_active": False}

# ==============================================================================
# --- 3. SEASONALITY LOGIC (Cyclical Encoding) ---
# ==============================================================================
def get_cyclical_seasonality():
    """
    Encodes date into Sine/Cosine and detects major market cycles:
    1. Q4 Retail (Christmas)
    2. Earnings Season (Quarterly reporting months)
    3. Summer Lull (Low volume/vacation months)
    4. Back to School (August/Sept retail boost)
    """
    today = datetime.now()
    month = today.month
    
    # 1. Math Encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # 2. Cycle Detection
    # Q4 Retail: Oct, Nov, Dec
    is_q4_retail = 1 if month in [10, 11, 12] else 0
    
    # Earnings Season: Usually the month after quarter end (Jan, Apr, Jul, Oct)
    is_earnings_season = 1 if month in [1, 4, 7, 10] else 0
    
    # Summer Lull: June, July, August (Vacation season, lower volume)
    is_summer_lull = 1 if month in [6, 7, 8] else 0
    
    # Back to School: August, September
    is_back_to_school = 1 if month in [8, 9] else 0
    
    return {
        "seasonality_sin": month_sin,
        "seasonality_cos": month_cos,
        "is_q4_retail": is_q4_retail,
        "is_earnings_season": is_earnings_season,
        "is_summer_lull": is_summer_lull,
        "is_back_to_school": is_back_to_school
    }
    
# ==============================================================================
# --- 4. MASTER FETCH FUNCTION ---
# ==============================================================================
def fetch_macro_context():
    """
    Master function to get the Global State Vector.
    Returns a dictionary with Interest Rates, Sahm Signal, and Seasonality.
    """
    print("📡 Fetching Macro Context...")
    
    sahm_data = calculate_sahm_rule()
    seasonality = get_cyclical_seasonality()
    
    # Fetch Interest Rate (Fed Funds)
    fed_funds = 5.33 # Default fallback
    if fred:
        try:
            # FEDFUNDS is monthly, might need DFF (Daily) for quicker updates, 
            # but FEDFUNDS is the standard policy rate benchmark.
            data = fred.get_series('FEDFUNDS', limit=5) 
            if not data.empty:
                fed_funds = float(data.iloc[-1])
        except Exception as e:
            print(f"Error fetching Fed Funds: {e}")
        
    global_context = {
        "interest_rate": fed_funds,
        **sahm_data,
        **seasonality
    }
    
    print(f"✅ Macro Context Loaded: {global_context}")
    return global_context

# ==============================================================================
# --- MAIN EXECUTION (For Testing) ---
# ==============================================================================
if __name__ == "__main__":
    # This block only runs if you run `python macro_ingest.py` directly
    context = fetch_macro_context()
    print("\n--- FINAL OUTPUT ---")
    print(json.dumps(context, indent=4))