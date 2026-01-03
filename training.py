import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
FEATURES = [
        'Close', 'MA_10', 'MA_30',
        'Volume', 'Daily_Range', 'Return_1d',
        'Log_Return_1d', 'Momentum_5', 'Momentum_10',
        'RSI_14', 'MACD', 'MACD_Signal'
]
TARGET = 'Next_Close'


def load_data(train_path, test_path):
    """Loads the training and testing datasets from CSV files."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: One or both data files ({train_path}, {test_path}) not found.")
        print("Please run 'data_preparation.py' first.")
        return None, None

    print("1. Loading data from CSV files.")
    COL_NAMES = ['Date', TARGET] + FEATURES
    df_train = pd.read_csv(train_path,
                           index_col=0,
                           parse_dates=True,
                           skiprows=3,
                           header=None,
                           names=COL_NAMES)

    df_test = pd.read_csv(test_path,
                          index_col=0,
                          parse_dates=True,
                          skiprows=3,
                          header=None,
                          names=COL_NAMES)

    print(f"Train samples loaded: {len(df_train)}")
    print(f"Test samples loaded: {len(df_test)}")

    # Ensure all required features and the target exist
    if not all(col in df_train.columns for col in FEATURES + [TARGET]):
        print("Error: Missing required columns in loaded data. Check FEATURES and TARGET lists.")
        return None, None

    return df_train, df_test

def train_and_evaluate(df_train, df_test, features, target):
    """Trains a Linear Regression model and evaluates its performance."""

    # Prepare data for sklearn
    X_train = df_train[features].values
    y_train = df_train[target].values

    X_test = df_test[features].values
    y_test = df_test[target].values

    print("2. Training Linear Regression Baseline Model (sklearn).")

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    print("3. Model Evaluation:")

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Features used: {', '.join(features)}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} USD")
    print(f"R-squared (R2) Score: {r2:.4f} (Closer to 1.0 is better)")


if __name__ == '__main__':
    # Execute the loading and training workflow
    df_train_data, df_test_data = load_data(TRAIN_FILE, TEST_FILE)

    if df_train_data is not None:
        train_and_evaluate(df_train_data, df_test_data, FEATURES, TARGET)
