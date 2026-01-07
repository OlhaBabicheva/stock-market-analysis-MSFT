import os # Standard library for interacting with the operating system (checking file paths)
import pandas as pd # Data manipulation library
import numpy as np # Numerical library (used for mathematical operations)
from sklearn.linear_model import LinearRegression, Ridge  # Ridge adds L2 regularization to Linear Regression
from sklearn.ensemble import RandomForestRegressor # A bagging-based ensemble model (often more stable than boosting)
from sklearn.svm import SVR # Support Vector Regression (good for non-linear patterns)
from sklearn.neural_network import MLPRegressor # Multi-layer Perceptron (Neural Network) regressor
from sklearn.preprocessing import StandardScaler # Crucial for models like SVR and Ridge
from sklearn.metrics import mean_squared_error, r2_score # Functions to calculate model accuracy

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
    """Trains multiple models to predict the Next_Close price."""

    # Prepare data for sklearn
    X_train_raw = df_train[features].values
    y_train = df_train[target].values

    X_test_raw = df_test[features].values
    y_test = df_test[target].values

    # Scaling is mandatory for SVR and Ridge to perform correctly
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=10.0), # Penalizes large coefficients to reduce noise sensitivity
        "Random Forest": RandomForestRegressor(
            n_estimators=400, # Number of trees
            random_state=42
        ), # Stable bagging
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(10, 64, 64), # Three hidden layers with 10, 64 and 64 neurons
            activation='relu',  # Rectified Linear Unit activation function
            solver='adam',      # Optimizer for weight optimization
            max_iter=1000,      # Maximum number of iterations
            random_state=42     # For reproducible results
        ),
        "SVR (RBF Kernel)": SVR(
            kernel='linear',
            epsilon=0.1 # Threshold where no penalty is given to errors
        ), # Non-linear boundary mapping
    }

    results = []
    print(f"2. Training models to predict absolute {target}...")

    for name, model in models.items():
        # Training directly on the absolute price
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "Model": name,
            "MSE": round(mse, 4),
            "RMSE (USD)": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })

    comparison_df = pd.DataFrame(results)
    print("="*50)
    print(f"MODEL COMPARISON SUMMARY (Target: {target})")
    print("="*50)
    print(comparison_df.sort_values(by="RMSE (USD)").to_string(index=False))
    print("="*50)

    return comparison_df


if __name__ == '__main__':
    # Execute the loading and training workflow
    df_train_data, df_test_data = load_data(TRAIN_FILE, TEST_FILE)

    if df_train_data is not None:
        train_and_evaluate(df_train_data, df_test_data, FEATURES, TARGET)
