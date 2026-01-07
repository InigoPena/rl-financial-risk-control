import yfinance as yf
import pandas as pd
import os

# Settings
START_DATE = "2000-11-08"
END_DATE   = "2025-11-14"
SYMBOL     = "DX-Y.NYB" # Dollar Index
OUTPUT_FILE = "usd_yfinance_2000_2025.csv"

def download_usd_data():
    print(f"Downloading {SYMBOL} from {START_DATE} to {END_DATE} using yfinance...")
    
    # Download data
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval="1d")
    
    if df.empty:
        print("Error: No data downloaded. Check symbol or internet connection.")
        return

    # Flatten columns if MultiIndex (yfinance often returns MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index to make Date a column
    df.reset_index(inplace=True)
    
    # Ensure Date format
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Data saved to {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    download_usd_data()
