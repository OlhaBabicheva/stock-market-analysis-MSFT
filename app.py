import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta

# --- APP CONFIGURATION ---
TICKER = 'MSFT'
MODEL_PATH = 'models_bundle.joblib'

st.set_page_config(page_title="MSFT Price Predictor", layout="wide")

@st.cache_resource
def load_bundle():
    """Loads the saved models and scaler."""
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

def get_latest_data(ticker):
    """Fetches and prepares the most recent data for prediction."""
    # Fetch enough data to calculate technical indicators (approx 60 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    try:
        data = yf.download(ticker, start=start_date, end=None, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty:
        return None

    # Feature Engineering (Mirrors data_preparation.py)
    # Get the Close prices
    close_prices = data['Close'].copy()

    # Feature 1 & 2: Simple Moving Averages
    data['MA_10'] = close_prices.rolling(window=10).mean()
    data['MA_30'] = close_prices.rolling(window=30).mean()

    # Feature 3: Daily Range (Measure of volatility)
    data['Daily_Range'] = data['High'] - data['Low']

    # Feature 4 & 5: Daily Returns
    data['Return_1d'] = close_prices.pct_change()
    data['Log_Return_1d'] = np.log(close_prices / close_prices.shift(1))

    # Feature 6 & 7: Momentum
    data['Momentum_5'] = close_prices - close_prices.shift(5)
    data['Momentum_10'] = close_prices - close_prices.shift(10)

    # Feature 8: RSI (Relative Strength Index)
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # Feature 9 & 10: MACD Indicator
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data.dropna()

# --- UI LAYOUT ---
st.title("📈 Microsoft (MSFT) Stock Price Predictor")
st.markdown("This app provides statistical predictions for the **Next Day Closing Price** using pre-trained ML models.")

bundle = load_bundle()

if bundle is None:
    st.error("Model bundle not found. Please run `training.py` first to generate `models_bundle.joblib`.")
else:
    latest_df = get_latest_data(TICKER)

    if latest_df is not None:
        # Get the very last row for prediction
        current_data = latest_df.iloc[[-1]]
        last_date = current_data.index[0].strftime('%Y-%m-%d')
        last_price = float(current_data['Close'].iloc[0])

        # Summary Statistics Row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Market Status")
            st.metric("Last Known Close", f"${last_price:,.2f}")
            st.caption(f"Data as of: {last_date}")

        with col2:
            st.subheader("Model Selection")
            selected_model_name = st.selectbox("Choose Model", list(bundle['models'].keys()))

        with col3:
            st.subheader("Volatility Index")
            daily_range = float(current_data['Daily_Range'].iloc[0])
            st.metric("Intraday Range", f"${daily_range:.2f}")

        # Prediction Logic
        features_to_use = bundle['features']
        X_raw = current_data[features_to_use].values
        X_scaled = bundle['scaler'].transform(X_raw)

        model = bundle['models'][selected_model_name]
        prediction_val = model.predict(X_scaled)
        prediction = float(prediction_val[0])

        st.divider()

        # Display Prediction Statistics
        st.subheader(f"Statistical Analysis: {selected_model_name}")
        diff = prediction - last_price
        percent = (diff / last_price) * 100

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label="Predicted Next Close", 
                value=f"${prediction:,.2f}", 
                delta=f"{diff:+.2f} ({percent:+.2f}%)"
            )

        with res_col2:
            sentiment = "Bullish" if diff > 0 else "Bearish"
            st.info(f"Signal: **{sentiment}** | Expected change of **{percent:.2f}%**")

        # Raw Data View
        with st.expander("View Raw Feature Data (Last 10 Days)"):
            st.dataframe(latest_df.tail(10))

    else:
        st.error("Failed to fetch recent data from Yahoo Finance.")

st.sidebar.info("Disclaimer: Statistical models are for informational purposes only. Trading involves significant risk.")