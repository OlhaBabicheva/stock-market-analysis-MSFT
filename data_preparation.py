import yfinance as yf # Yahoo Finance API downloader (fetches historical market data)
import numpy as np # Numerical library (used for mathematical operations)

# --- CONFIGURATION ---
TICKER = 'MSFT'
START_DATE = '2019-01-01'
TEST_SIZE_RATIO = 0.2
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

def download_and_prepare_data(ticker, start_date, test_ratio):
    """
    Downloads historical stock data, performs feature engineering, and splits it 
    chronologically into training and testing sets.
    """
    print(f"1. Downloading historical data for {ticker} (Start Date: {start_date})...")

    try:
        data = yf.download(ticker, start=start_date, end=None, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty:
        print("No data retrieved. Exiting data pipeline.")
        return None

    print(f"Download successful. Total samples: {len(data)}")
    print("2. Feature Engineering: Calculating Moving Averages and setting target...")

    # Get the Close prices
    close_prices = data['Close'].copy()

    # Feature 1 & 2: Simple Moving Averages
    data['MA_10'] = close_prices.rolling(window=10).mean()
    data['MA_30'] = close_prices.rolling(window=30).mean()

    # Feature 3: Daily Range (Measure of volatility)
    data['Daily_Range'] = data['High'] - data['Low']

    # Feature 4 & 5: Daily Returns
    #(returns capture day to day relative price changes)
    data['Return_1d'] = close_prices.pct_change()
    # Log returns are time-additive and commonly assumed to be closer to normality
    data['Log_Return_1d'] = np.log(close_prices / close_prices.shift(1))

    # Feature 6 & 7: Momentum
    # (measures medium-term trend persistence.
    # Positive values indicate upward pressure, negative values downward pressure)
    data['Momentum_5'] = close_prices - close_prices.shift(5)
    data['Momentum_10'] = close_prices - close_prices.shift(10)

    # Feature 8: RSI (Relative Strength Index)
    # RSI measures the speed and magnitude of recent price movements
    # Values >70 often indicate overbought conditions, Values <30 often indicate oversold conditions

    delta = close_prices.diff()
    # Separate positive and negative price changes
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Rolling averages of gains and losses (standard 14-day window)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    # Relative Strength and RSI formula
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # Feature 9 & 10: MACD Indicator - trend-following momentum indicator
    # (MACD captures the relationship between short-term and long-term trends)

    # Short-term exponential moving average
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    # Long-term exponential moving average
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    # MACD line
    data['MACD'] = ema_12 - ema_26
    # Signal line (EMA of MACD)
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Target Variable: Predict the next day's Close price (Next_Close)
    data['Next_Close'] = close_prices.shift(-1)

    # Define the features we will use for the model
    FEATURES = [
        'Close', 'MA_10', 'MA_30',
        'Volume', 'Daily_Range', 'Return_1d',
        'Log_Return_1d', 'Momentum_5', 'Momentum_10',
        'RSI_14', 'MACD', 'MACD_Signal'
    ]

    # Drop rows that have NaN values due to the initial 30-day rolling
    # window and the last target row
    df = data.dropna()

    print(f"Data cleaned. Usable data points: {len(df)}")

    # Calculate the index for the chronological split
    train_size = int(len(df) * (1 - test_ratio))

    # Split the dataframes
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    return df_train, df_test, FEATURES

def save_data(df_train, df_test, features):
    """Saves the train and test DataFrames, including the target and features, to CSV."""

    # Columns to save: Target & Features
    cols_to_save = ['Next_Close'] + features

    # Save to CSV files
    df_train[cols_to_save].to_csv(TRAIN_FILE)
    df_test[cols_to_save].to_csv(TEST_FILE)

    print("3. Data split and saved to CSV files:")
    print(f"Training set size: {len(df_train)} samples ({TRAIN_FILE})")
    print(f"Testing set size: {len(df_test)} samples ({TEST_FILE})")

    return features


if __name__ == '__main__':
    data_output = download_and_prepare_data(TICKER, START_DATE, TEST_SIZE_RATIO)

    if data_output:
        df_train, df_test, features_list = data_output
        save_data(df_train, df_test, features_list)
