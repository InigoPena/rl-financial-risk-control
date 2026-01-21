"""
SafePolicy-RM: Integrated Risk Management Agent for US Treasury Bonds Trading
Data Download and Feature Engineering Script

This script downloads US Treasury 10-Year yield data from Yahoo Finance (2015-present)
and calculates risk management indicators for RL training.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_volatility(df, window=20):
    """Calculate rolling volatility (std of returns)"""
    returns = df['Close'].pct_change()
    return returns.rolling(window=window).std()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_drawdown(df):
    """Calculate current drawdown from peak"""
    cumulative = (1 + df['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_roc(df, period=10):
    """Calculate Rate of Change"""
    roc = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    return roc

def calculate_ma_slope(df, window=20):
    """Calculate slope of moving average"""
    ma = df['Close'].rolling(window=window).mean()
    # Slope as percentage change over the window
    slope = (ma - ma.shift(5)) / ma.shift(5) * 100
    return slope

def download_and_process_treasury_data(start_date='2015-01-01', end_date=None):
    """
    Download US Treasury 10-Year yield data and calculate all indicators
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with all indicators
    """
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading US Treasury 10-Year Yield (^TNX) data from {start_date} to {end_date}...")
    
    # Download US Treasury 10-Year Yield (^TNX)
    # Alternative tickers: TLT (iShares 20+ Year Treasury Bond ETF), IEF (7-10 Year Treasury ETF)
    treasury = yf.download('^TNX', start=start_date, end=end_date, progress=False)
    
    if treasury.empty:
        print("No data found for ^TNX, trying TLT (Treasury Bond ETF)...")
        treasury = yf.download('TLT', start=start_date, end=end_date, progress=False)
    
    print(f"Downloaded {len(treasury)} rows of data")
    
    # Create feature dataframe
    df = pd.DataFrame(index=treasury.index)
    df['Close'] = treasury['Close']
    df['High'] = treasury['High']
    df['Low'] = treasury['Low']
    df['Open'] = treasury['Open']
    df['Volume'] = treasury['Volume']
    
    print("\nCalculating indicators...")
    
    # ===========================
    # 1. RISK INDICATORS
    # ===========================
    print("  - Risk indicators...")
    df['VOLATILITY'] = calculate_volatility(df, window=20)
    df['ATR'] = calculate_atr(df, period=14)
    df['DRAWDOWN'] = calculate_drawdown(df)
    df['RETURN_1D'] = df['Close'].pct_change(1)
    df['RETURN_5D'] = df['Close'].pct_change(5)
    
    # ===========================
    # 2. TREND INDICATORS
    # ===========================
    print("  - Trend indicators...")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA20/CLOSE'] = df['MA20'] / df['Close']
    df['MA50/CLOSE'] = df['MA50'] / df['Close']
    df['SLOPE_MA20'] = calculate_ma_slope(df, window=20)
    
    # ===========================
    # 3. MOMENTUM INDICATORS
    # ===========================
    print("  - Momentum indicators...")
    df['RSI_14'] = calculate_rsi(df, period=14)
    df['ROC_10'] = calculate_roc(df, period=10)
    
    # ===========================
    # 4. PRICE CONTEXT
    # ===========================
    # CLOSE already included (represents yield %)
    
    # Drop rows with NaN values (due to rolling calculations)
    df_clean = df.dropna()
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index[0]} to {df_clean.index[-1]}")
    
    return df_clean

def save_data(df, filename='treasury_data_safepolicy.parquet'):
    """Save processed data to Parquet"""
    df.to_parquet(filename, engine='pyarrow', compression='snappy')
    print(f"\nData saved to {filename}")

def display_summary(df):
    """Display summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS - US TREASURY 10-YEAR YIELD")
    print("="*70)
    
    print("\n📊 RISK INDICATORS:")
    print(f"  Volatility (mean):        {df['VOLATILITY'].mean():.4f}")
    print(f"  ATR (mean):               {df['ATR'].mean():.2f}")
    print(f"  Max Drawdown:             {df['DRAWDOWN'].min():.2%}")
    print(f"  1D Return (mean):         {df['RETURN_1D'].mean():.4f}")
    print(f"  5D Return (mean):         {df['RETURN_5D'].mean():.4f}")
    
    print("\n📈 TREND INDICATORS:")
    print(f"  MA20/Close (mean):        {df['MA20/CLOSE'].mean():.4f}")
    print(f"  MA50/Close (mean):        {df['MA50/CLOSE'].mean():.4f}")
    print(f"  MA20 Slope (mean):        {df['SLOPE_MA20'].mean():.4f}")
    
    print("\n⚡ MOMENTUM INDICATORS:")
    print(f"  RSI(14) (mean):           {df['RSI_14'].mean():.2f}")
    print(f"  ROC(10) (mean):           {df['ROC_10'].mean():.2f}")
    
    print("\n📋 YIELD CONTEXT:")
    print(f"  10Y Yield (latest):       {df['Close'].iloc[-1]:.2f}%")
    print(f"  10Y Yield (min):          {df['Close'].min():.2f}%")
    print(f"  10Y Yield (max):          {df['Close'].max():.2f}%")
    
    print("\n" + "="*70)

def get_feature_columns():
    """Return list of feature columns for RL training"""
    features = [
        # Risk
        'VOLATILITY', 'ATR', 'DRAWDOWN', 'RETURN_1D', 'RETURN_5D',
        # Trend
        'MA20/CLOSE', 'MA50/CLOSE', 'SLOPE_MA20',
        # Momentum
        'RSI_14', 'ROC_10',
        # Price Context
        'CLOSE'
    ]
    return features

if __name__ == "__main__":
    print("="*70)
    print("SafePolicy-RM: US Treasury Bonds Trading Risk Management Agent")
    print("Data Preparation Script")
    print("="*70)
    
    # Download and process data
    df = download_and_process_treasury_data(start_date='2015-01-01')
    
    # Display summary
    display_summary(df)
    
    # Save to Parquet
    save_data(df, 'treasury_data_safepolicy.parquet')
    
    # Show feature columns for RL
    print("\n🤖 FEATURES FOR RL TRAINING:")
    features = get_feature_columns()
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    
    print("\n✅ Data preparation complete!")
    print("\nNext steps:")
    print("  1. Load 'treasury_data_safepolicy.parquet' in your gym-trading environment")
    print("  2. Use these features as observations for your RL agent")
    print("  3. Train your agent to optimize position sizing, stop-loss, and take-profit")
    print("\n💡 Note: ^TNX represents the yield percentage (e.g., 4.09 = 4.09%)")
    print("    Higher yields typically mean lower bond prices and vice versa")
    print("\n" + "="*70)