import pandas as pd
import requests
import datetime
from io import StringIO
import os

def patch_csv_sectors():
    csv_file = "live_risk_scores.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Could not find {csv_file}")
        return

    print(f"📂 Loading {csv_file}...")
    df_csv = pd.read_csv(csv_file)
    
    # 1. Fetch S&P 500 Data (The "Truth" Source)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    print(f"📡 Fetching Sector data from Wikipedia...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        df_wiki = None
        for t in tables:
            if "Symbol" in t.columns and "GICS Sector" in t.columns:
                df_wiki = t
                break
        
        if df_wiki is None:
            print("❌ Could not download Sector data.")
            return

        # Map Ticker -> Sector
        sector_map = dict(zip(df_wiki['Symbol'], df_wiki['GICS Sector']))
        sector_map.update(dict(zip(df_wiki['Symbol'].str.replace('.', '-'), df_wiki['GICS Sector'])))
        
        # Manual Overrides for non-S&P stocks
        manual_sectors = {
            "IONQ": "Technology", "SNAP": "Communication Services",
            "SIRI": "Communication Services", "COR": "Health Care",
            "CPAY": "Financials", "DOC": "Real Estate",
            "KKR": "Financials", "GE": "Industrials", "UBER": "Industrials"
        }
        sector_map.update(manual_sectors)

        # 2. Apply the Patch
        print("🛠️  Patching 'Discovered' sectors...")
        df_csv['Sector'] = df_csv['Ticker'].map(sector_map).fillna(df_csv['Sector'])

        # 3. Fix Timestamps if missing
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if 'Last_Updated' in df_csv.columns:
            df_csv['Last_Updated'] = df_csv['Last_Updated'].fillna(now_str)
        else:
            df_csv['Last_Updated'] = now_str

        # 4. Save
        df_csv.to_csv(csv_file, index=False)
        print(f"✅ Success! Patched {len(df_csv)} records.")
        print(df_csv[['Ticker', 'Sector']].head())

    except Exception as e:
        print(f"❌ Error patching CSV: {e}")

if __name__ == "__main__":
    patch_csv_sectors()