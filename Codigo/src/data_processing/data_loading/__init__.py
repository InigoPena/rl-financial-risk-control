from .utils import load_dataset as _load_dataset


# Load FOREX datasets
FOREX_EURUSD_1H_ASK = _load_dataset('FOREX_EURUSD_1H_ASK', 'Time')

# Load Stocks datasets
STOCKS_GOOGL = _load_dataset('STOCKS_GOOGL', 'Date')

# Load Gold and Hedge datasets (para GoldHedgeEnv)
GOLD_DATA = _load_dataset('gold_data', 'Date')
HEDGE_DATA = _load_dataset('treasury_data_safepolicy', 'Date')
