import yfinance as yf
import pandas as pd

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

    # Target Variable: Predict the next day's Close price (Next_Close)
    data['Next_Close'] = close_prices.shift(-1)

    # Define the features we will use for the model
    FEATURES = ['Close', 'MA_10', 'MA_30', 'Volume', 'Daily_Range']

    # Drop rows that have NaN values (due to the initial 30-day rolling window and the last target row)
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

    print(f"3. Data split and saved to CSV files:")
    print(f"Training set size: {len(df_train)} samples ({TRAIN_FILE})")
    print(f"Testing set size: {len(df_test)} samples ({TEST_FILE})")

    return features


if __name__ == '__main__':
    data_output = download_and_prepare_data(TICKER, START_DATE, TEST_SIZE_RATIO)

    if data_output:
        df_train, df_test, features_list = data_output
        save_data(df_train, df_test, features_list)
