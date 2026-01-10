# MSFT Stock Price Predictor

This machine learning project is designed to predict the next day's closing price for Microsoft (MSFT) stock. It includes a complete workflow from data acquisition and feature engineering to model training and real-time prediction via a web interface.

## Project Overview

The system is divided into three main components:

1. Data Preparation (`data_preparation.py`): Fetches historical data from Yahoo Finance and calculates technical indicators (MA, RSI, MACD, etc.).

2. Model Training (`training.py`): Trains multiple regression models (Linear Regression, Random Forest, Neural Networks, SVR) and saves the trained models into a model bundle.

3. Streamlit Dashboard (`app.py`): A user-friendly interface that pulls live market data and displays predictions based on the trained models.

## How to Run the Project

To successfully run the app, you must follow the pipeline order to ensure the necessary data and models exist.

### Step 1: Install Dependencies

Ensure you have the required Python libraries installed:

`pip install -r requirements.txt`

### Step 2: Prepare the Data

Run the data preparation script to download historical prices and generate the training/testing CSV files.

`python data_preparation.py`

### Step 3: Train the Models

Run the training script to train and evaluate the models. This will generate a file named models_bundle.joblib, which the dashboard requires.

`python training.py`

### Step 4: Launch the Streamlit App

Once the models_bundle.joblib file is generated, you can launch the interactive dashboard:

`streamlit run app.py`

***Disclaimer:*** This tool is for educational purposes only. Stock market trading involves significant risk, and past performance is not indicative of future results.