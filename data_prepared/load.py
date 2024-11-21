import pandas as pd
import numpy as np
import os
from functools import reduce

def load_stock_data(ticker):
    """
    Load stock data for the specified ticker and return training and testing datasets.
    """
    # Define file paths
    file_path = f"data/{ticker}.csv"

    # Load individual datasets
    stock_return_df = load_stock_return(file_path)
    bid_ask_spread_df = load_bid_ask_spread(file_path)
    volume_df = trading_volume(file_path)

    # Reset index for merging
    stock_return_df.reset_index(inplace=True)
    bid_ask_spread_df.reset_index(inplace=True)
    volume_df.reset_index(inplace=True)

    # Merge datasets on 'Date'
    dfs = [stock_return_df, bid_ask_spread_df, volume_df]
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)

    # Split into training and testing datasets
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])  # Ensure 'Date' is datetime
    training_dataset = merged_df[merged_df["Date"] < "2023-01-02"]
    testing_dataset = merged_df[merged_df["Date"] >= "2023-01-02"]

    return training_dataset, testing_dataset

def load_stock_return(file_path):
    # Similar implementation as before
    stock_data = pd.read_csv(file_path, index_col=0)
    stock_data.index.name = "Date"
    stock_data.reset_index(inplace=True)

    if 'close' not in stock_data.columns:
        raise ValueError("The dataset must contain a 'close' column for stock prices.")

    stock_data['log_return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))
    stock_data.dropna(inplace=True)

    return stock_data[['Date', 'log_return']]

def load_bid_ask_spread(file_path):
    # Similar implementation as before
    stock_data = pd.read_csv(file_path, index_col=0)
    stock_data.index.name = 'Date'
    stock_data.reset_index(inplace=True)

    stock_data['high_shift'] = stock_data['high'].shift(1)
    stock_data['low_shift'] = stock_data['low'].shift(1)

    stock_data['beta'] = (np.log(stock_data['high'] * stock_data['low_shift'] / (stock_data['low'] * stock_data['high_shift']))) ** 2
    stock_data['gamma'] = (np.log(stock_data['high'] / stock_data['low']) ** 2 + np.log(stock_data['high_shift'] / stock_data['low_shift']) ** 2) / 2
    stock_data['alpha'] = stock_data['beta'] / stock_data['gamma']
    stock_data['Spread'] = 2 * (np.sqrt(np.exp(stock_data['alpha']) - 1))

    return stock_data[['Date', 'Spread']]

def trading_volume(file_path):
    # Similar implementation as before
    stock_data = pd.read_csv(file_path, index_col=0)
    stock_data.index.name = 'Date'
    stock_data.reset_index(inplace=True)

    return stock_data[['Date', 'volume']]

if __name__ == "__main__":
    # Process for UBS, DB, and MS
    tickers = ["UBS", "DB", "MS"]

    datasets = {}
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        train, test = load_stock_data(ticker)
        datasets[ticker] = {"train": train, "test": test}

    # Example: Access training and testing datasets for UBS
    ubs_train = datasets["UBS"]["train"]
    ubs_test = datasets["UBS"]["test"]

    # Optional: Save datasets to CSV for later use
    for ticker, data in datasets.items():
        data["train"].to_csv(f"output/{ticker}_training_dataset.csv", index=False)
        data["test"].to_csv(f"output/{ticker}_testing_dataset.csv", index=False)
