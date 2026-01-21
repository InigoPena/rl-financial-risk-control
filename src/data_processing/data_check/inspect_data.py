import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

print("### GOLD DATA ###")
try:
    gold_df = pd.read_parquet('Codigo/data/gold_data.parquet')
    print("Columns:", list(gold_df.columns))
    print("Shape:", gold_df.shape)
    # Check for specific columns
    has_ret = 'RETURN_1D' in gold_df.columns
    has_vol = 'VOLATILITY' in gold_df.columns
    print(f"Has RETURN_1D: {has_ret}")
    print(f"Has VOLATILITY: {has_vol}")
except Exception as e:
    print(f"Error: {e}")

print("\n### USD DATA ###")
try:
    usd_df = pd.read_parquet('Codigo/data/usd_data.parquet')
    print("Columns:", list(usd_df.columns))
    print("Shape:", usd_df.shape)
    # Check for specific columns
    has_ret = 'RETURN_1D' in usd_df.columns
    has_vol = 'VOLATILITY' in usd_df.columns
    print(f"Has RETURN_1D: {has_ret}")
    print(f"Has VOLATILITY: {has_vol}")
except Exception as e:
    print(f"Error: {e}")
