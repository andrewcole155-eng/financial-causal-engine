import pandas as pd
import json
import requests
from io import StringIO 

def update_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print(f"Fetching data from {url}...")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error fetching data: Status {response.status_code}")
            return

        # Load all tables
        tables = pd.read_html(StringIO(response.text))
        df = None
        
        # Find the main S&P 500 table
        for table in tables:
            if "Security" in table.columns and "GICS Sector" in table.columns:
                df = table
                break
        
        if df is None:
            print("❌ Could not identify the S&P 500 table.")
            return

        # Determine symbol column name
        symbol_col = "Symbol"
        if "Symbol" not in df.columns:
            for col in df.columns:
                if "Symbol" in col or "Ticker" in col:
                    symbol_col = col
                    break

        # 1. Create the Base Dictionary from Wikipedia (Live Data)
        # We store BOTH 'BRK.B' and 'BRK-B' to be safe
        ticker_lookup = dict(zip(df[symbol_col], df['Security']))
        
        # Add the hyphenated versions too (standard API format)
        ticker_lookup.update(dict(zip(df[symbol_col].str.replace('.', '-'), df['Security'])))

        # 2. INJECT MISSING / HISTORICAL / REBRANDED COMPANIES
        # Updated to remove delisted stocks and add new ticker symbols
        manual_fix = {
            # --- Recent Rebrands (The "Zombies" Fixed) ---
            "COR": "Cencora (formerly AmerisourceBergen)",
            "DOC": "Healthpeak Properties (formerly PEAK)",
            "EG": "Everest Group (formerly RE)",
            "DAY": "Dayforce (formerly Ceridian/CDAY)",
            "CPAY": "Corpay (formerly FLEETCOR/FLT)",
            "SW": "Smurfit WestRock (formerly WestRock)",
            
            # --- Special Handling / Niche Tickers ---
            "BRK.B": "Berkshire Hathaway (Class B)",
            "BF.B": "Brown-Forman Corp (Class B)",
            "IONQ": "IonQ, Inc.", # Non-S&P 500 (Quantum Tech)
            "SNAP": "Snap Inc.", # Non-S&P 500
            "SIRI": "Sirius XM Holdings", # Nasdaq 100
            
            # --- Standard Fixes ---
            "AAL": "American Airlines Group",
            "AAP": "Advance Auto Parts",
            "ALK": "Alaska Air Group",
            "ANSS": "ANSYS Inc.",
            "BBWI": "Bath & Body Works, Inc.",
            "BIO": "Bio-Rad Laboratories",
            "BWA": "BorgWarner Inc.",
            "CE": "Celanese Corp",
            "CMA": "Comerica Inc.",
            "CTLT": "Catalent, Inc.",
            "CZR": "Caesars Entertainment",
            "DFS": "Discover Financial Services",
            "DXC": "DXC Technology",
            "EMN": "Eastman Chemical",
            "ENPH": "Enphase Energy",
            "ETSY": "Etsy, Inc.",
            "FI": "Fiserv, Inc.",
            "FMC": "FMC Corporation",
            "HES": "Hess Corporation",
            "ILMN": "Illumina, Inc.",
            "JNPR": "Juniper Networks",
            "KMX": "CarMax, Inc.",
            "LNC": "Lincoln National Corp",
            "MKTX": "MarketAxess",
            "MRO": "Marathon Oil",
            "OGN": "Organon & Co.",
            "PARA": "Paramount Global",
            "QRVO": "Qorvo, Inc.",
            "RHI": "Robert Half Inc.",
            "SEE": "Sealed Air Corp",
            "TFX": "Teleflex Inc.",
            "VFC": "VF Corporation",
            "WBA": "Walgreens Boots Alliance",
            "WHR": "Whirlpool Corporation",
            "XRAY": "Dentsply Sirona",
            "ZION": "Zions Bancorporation"
        }

        # Merge the manual fix into the main dictionary
        print(f"Injecting {len(manual_fix)} manual overrides (fixes & non-S&P)...")
        ticker_lookup.update(manual_fix)

        # 3. Save to JSON
        output_file = 'sp500_companies.json'
        with open(output_file, 'w') as f:
            json.dump(ticker_lookup, f, indent=4)
        
        print(f"Success! Created {output_file} with {len(ticker_lookup)} companies.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_sp500_data()