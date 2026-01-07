import pandas as pd
import os

print("### USD DATA CHECK ###")
try:
    if not os.path.exists('Codigo/data/treasury_data.parquet'):
        print("File not found!")
    else:
        usd_df = pd.read_parquet('Codigo/data/treasury_data.parquet')
        print(f"Columns: {list(usd_df.columns)}")
        print(f"Has RETURN_1D: {'RETURN_1D' in usd_df.columns}")
        print(f"Has VOLATILITY: {'VOLATILITY' in usd_df.columns}")
        
        print("Columns:", list(usd_df.columns))
        print("Shape:", usd_df.shape)
except Exception as e:
    print(f"Error: {e}")
