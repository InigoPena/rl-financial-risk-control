import pandas as pd
import os

def check_alignment():
    base_dir = r"Codigo/data"
    gold_path = os.path.join(base_dir, 'gold_data.csv')
    usd_path = os.path.join(base_dir, 'usd_yfinance_2000_2025.csv')
    
    print(f"--- Checking Gold Data ({gold_path}) ---")
    if os.path.exists(gold_path):
        gold_df = pd.read_csv(gold_path, parse_dates=['Date'], index_col='Date')
        # gold_df.index = pd.to_datetime(gold_df.index) # Already done by parse_dates
        print(f"Shape: {gold_df.shape}")
        print(f"Start Date: {gold_df.index.min()}")
        print(f"End Date:   {gold_df.index.max()}")
    else:
        print("Gold file not found!")
        return

    print(f"\n--- Checking USD Data ({usd_path}) ---")
    if os.path.exists(usd_path):
        usd_df = pd.read_csv(usd_path)
        usd_df['Date'] = pd.to_datetime(usd_df['Date'])
        usd_df.set_index('Date', inplace=True)
        print(f"Shape: {usd_df.shape}")
        print(f"Start Date: {usd_df.index.min()}")
        print(f"End Date:   {usd_df.index.max()}")
    else:
        print("USD file not found!")
        return

    print(f"\n--- Intersection ---")
    common_index = gold_df.index.intersection(usd_df.index)
    print(f"Common Points: {len(common_index)}")
    print(f"Common Start: {common_index.min()}")
    print(f"Common End:   {common_index.max()}")

if __name__ == "__main__":
    check_alignment()
