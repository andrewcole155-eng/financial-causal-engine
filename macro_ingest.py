import json
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import os

# Load Configuration
def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()
fred = Fred(api_key=config['fred_api_key'])

def calculate_sahm_rule():
    """
    Fetches UNRATE, calculates 3-month moving average, 
    and compares to the 12-month low to detect recession.
    """
    try:
        # Fetch Unemployment Rate (UNRATE)
        unrate = fred.get_series('UNRATE')
        
        # 1. Calculate 3-month moving average
        unrate_3m_avg = unrate.rolling(window=3).mean()
        
        # 2. Calculate the lowest 3-month avg in the previous 12 months
        low_last_12m = unrate_3m_avg.rolling(window=12).min()
        
        # 3. Calculate Sahm Signal (Current 3m Avg - Lowest in last 12m)
        sahm_signal = unrate_3m_avg - low_last_12m
        
        # Get the latest value
        current_signal = sahm_signal.iloc[-1]
        
        # Boolean Switch: Sahm Rule triggers if signal >= 0.50
        is_recession = current_signal >= 0.50
        
        return {
            "sahm_signal_value": round(current_signal, 4),
            "is_recession_active": bool(is_recession),
            "unemployment_rate": unrate.iloc[-1]
        }
    except Exception as e:
        print(f"Error calculating Sahm Rule: {e}")
        return {"sahm_signal_value": 0.0, "is_recession_active": False}

def get_cyclical_seasonality():
    """
    Encodes the current date into Sine/Cosine features.
    This preserves the 'closeness' of December to January.
    """
    today = datetime.now()
    month = today.month
    day_of_year = today.timetuple().tm_yday
    
    # Mathematical encoding
    # sin_time = sin(2 * pi * month / 12)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Check for Retail Season (Q4)
    is_q4_retail = 1 if month in [10, 11, 12] else 0
    
    return {
        "seasonality_sin": month_sin,
        "seasonality_cos": month_cos,
        "is_q4_retail": is_q4_retail
    }

def fetch_macro_context():
    """
    Master function to get the Global State Vector.
    """
    sahm_data = calculate_sahm_rule()
    seasonality = get_cyclical_seasonality()
    
    # Fetch Interest Rate (Fed Funds)
    try:
        fed_funds = fred.get_series('FEDFUNDS').iloc[-1]
    except:
        fed_funds = 5.33 # Fallback
        
    global_context = {
        "interest_rate": fed_funds,
        **sahm_data,
        **seasonality
    }
    
    print(f"Global Macro Context: {global_context}")
    return global_context

if __name__ == "__main__":
    fetch_macro_context()