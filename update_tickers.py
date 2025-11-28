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

        # 1. Create the Base Dictionary from Wikipedia
        ticker_lookup = dict(zip(df[symbol_col], df['Security']))
        
        # Add hyphenated versions (BRK.B -> BRK-B)
        ticker_lookup.update(dict(zip(df[symbol_col].str.replace('.', '-'), df['Security'])))

        # 2. INJECT MACRO ASSETS (Crypto, Forex, Indices)
        additional_assets = {
            "I:SPX": "S&P 500 Index",
            "I:NDX": "Nasdaq 100 Index",
            "I:DJI": "Dow Jones Industrial Average",
            "I:VIX": "CBOE Volatility Index",
            "I:RUT": "Russell 2000 Index",
            "X:BTCUSD": "Bitcoin (USD)",
            "X:ETHUSD": "Ethereum (USD)",
            "X:SOLUSD": "Solana (USD)",
            "X:ADAUSD": "Cardano (USD)",
            "X:DOGEUSD": "Dogecoin (USD)",
            "X:LTCUSD": "Litecoin (USD)",
            "X:MATICUSD": "Polygon (USD)",
            "X:DOTUSD": "Polkadot (USD)",
            "C:EURUSD": "Euro / US Dollar",
            "C:USDJPY": "US Dollar / Japanese Yen",
            "C:GBPUSD": "British Pound / US Dollar",
            "C:AUDUSD": "Australian Dollar / US Dollar",
            "C:USDCAD": "US Dollar / Canadian Dollar",
            "C:USDCHF": "US Dollar / Swiss Franc",
            "C:NZDUSD": "New Zealand Dollar / US Dollar"
        }

        # 3. INJECT REBRANDS & CORRECTIONS
        manual_fix = {
            "COR": "Cencora (formerly AmerisourceBergen)",
            "DOC": "Healthpeak Properties (formerly PEAK)",
            "EG": "Everest Group (formerly RE)",
            "DAY": "Dayforce (formerly Ceridian/CDAY)",
            "CPAY": "Corpay (formerly FLEETCOR/FLT)",
            "SW": "Smurfit WestRock (formerly WestRock)",
            "BRK.B": "Berkshire Hathaway (Class B)",
            "BF.B": "Brown-Forman Corp (Class B)",
            "IONQ": "IonQ, Inc.", 
            "SNAP": "Snap Inc.", 
            "SIRI": "Sirius XM Holdings",
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

        # Merge Dictionaries
        ticker_lookup.update(additional_assets)
        ticker_lookup.update(manual_fix)

        # 4. CONVERT TO LIST OF OBJECTS [{"ticker": "X", "name": "Y"}, ...]
        formatted_data = [{"ticker": k, "name": v} for k, v in ticker_lookup.items()]

        # 5. Save to JSON (Using the filename your Worker expects)
        # We save to BOTH names to prevent future errors
        
        output_files = ['sp500_companies.json', 'sp500_map.json']
        
        for filename in output_files:
            with open(filename, 'w') as f:
                json.dump(formatted_data, f, indent=4)
            print(f"✅ Saved {filename} ({len(formatted_data)} items)")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    update_sp500_data()