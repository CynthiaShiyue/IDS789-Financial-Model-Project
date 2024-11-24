import pandas as pd
import numpy as np
import os
from functools import reduce


def load(verbose=False):
    # Load datasets
    ubs_stock_return = load_stock_return(ticker="UBS", verbose=verbose)
    ubs_Bid_Ask_Spread_df = load_bid_ask_spread("data/UBS.csv")
    ubs_volume_df = trading_volume("data/UBS.csv")
    vix_data = load_cboe_vix(file_path="data/^VIX.csv")
    eur_chf_data = load_eur_chf(file_path="data/EURCHF=X.csv")
    DB_stock_return = load_stock_return(ticker="DB", verbose=verbose)
    MS_stock_return = load_stock_return(ticker="MS", verbose=verbose)
    SPY_stock_return = load_stock_return(ticker="SPY", verbose=verbose)
    FTSE_stock_return = load_stock_return(ticker="^FTSE", verbose=verbose)
    oil_prices_df = load_oil_prices("data/CL=F.csv")
    gold_prices_df = load_gold_prices("data/GC=F.csv")

    oil_prices_df.reset_index(inplace=True)
    gold_prices_df.reset_index(inplace=True)

    # Combine all datasets

    dfs = [
        ubs_stock_return,
        ubs_Bid_Ask_Spread_df,
        ubs_volume_df,
        vix_data,
        eur_chf_data,
        oil_prices_df,
        gold_prices_df,
        DB_stock_return,
        MS_stock_return,
        SPY_stock_return,
        FTSE_stock_return,
    ]

    # Ensure all 'Date' columns are in datetime64[ns] format
    for df in dfs:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Merge datasets on 'Date'
    df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)

    # Split into training and testing datasets
    training_dataset = df[df["Date"] < "2023-01-02"]
    testing_dataset = df[df["Date"] >= "2023-01-02"]

    return training_dataset, testing_dataset


def load_stock_return(ticker="UBS", verbose=False):
    """
    Load stock data from a CSV file and calculate stock returns.

    Parameters:
        ticker (str): Stock ticker symbol. Defaults to "UBS".
        verbose (bool): If True, prints debug information. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with 'Date' and log returns.
    """
    file_path = f"data/{ticker}.csv"

    # Load the stock data
    try:
        stock_data = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file for ticker '{ticker}' was not found at {file_path}."
        )
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file for ticker '{ticker}' is empty.")

    # Ensure the index is a datetime format
    stock_data.index = pd.to_datetime(stock_data.index, errors="coerce")
    if stock_data.index.isna().any():
        raise ValueError("The index contains invalid or non-datetime values.")

    # Ensure the index is named "Date"
    if stock_data.index.name != "Date":
        stock_data.index.name = "Date"

    # Reset index to make 'Date' a column
    stock_data.reset_index(inplace=True)

    # Ensure the 'close' column is present and numeric
    if "close" not in stock_data.columns:
        raise ValueError("The dataset must contain a 'close' column for stock prices.")
    if not pd.api.types.is_numeric_dtype(stock_data["close"]):
        raise ValueError("The 'close' column must contain numeric values.")

    # Ensure there are enough unique values to calculate returns
    if stock_data["close"].nunique() <= 1:
        raise ValueError(
            "The 'close' column must have more than one unique value to calculate returns."
        )

    # Calculate the daily log stock returns
    stock_data[f"{ticker} log_return"] = np.log(
        stock_data["close"] / stock_data["close"].shift(1)
    )

    # Drop rows with NaN values resulting from the calculation
    stock_data.dropna(inplace=True)

    if verbose:
        print(
            f"Processed data for {ticker}: {len(stock_data)} rows remaining after cleaning."
        )

    return stock_data[["Date", f"{ticker} log_return"]]


def load_bid_ask_spread(file_path="data/UBS.csv"):
    """
    Calculate estimated spread from historical high and low prices in a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing historical data with 'Date', 'High', and 'Low' columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the 'Date' and calculated 'Spread'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load historical data from CSV
    stock_data = pd.read_csv(file_path, index_col=0)
    stock_data.index.name = "Date"  # Name the index as 'Date'
    stock_data.reset_index(inplace=True)  # Convert index to a column for merging

    # Use lowercase column names to match the CSV
    stock_data["high_shift"] = stock_data["high"].shift(1)
    stock_data["low_shift"] = stock_data["low"].shift(1)

    # Calculate beta and gamma
    stock_data["beta"] = (
        np.log(
            stock_data["high"]
            * stock_data["low_shift"]
            / (stock_data["low"] * stock_data["high_shift"])
        )
    ) ** 2
    stock_data["gamma"] = (
        np.log(stock_data["high"] / stock_data["low"]) ** 2
        + np.log(stock_data["high_shift"] / stock_data["low_shift"]) ** 2
    ) / 2

    # Calculate alpha and spread
    stock_data["alpha"] = stock_data["beta"] / stock_data["gamma"]
    stock_data["Spread"] = 2 * (np.sqrt(np.exp(stock_data["alpha"]) - 1))

    # Return only the 'Date' and 'Spread' columns
    result = stock_data[["Date", "Spread"]]

    return result


def trading_volume(file_path="data/UBS.csv"):
    """
    Extract trading volume from historical stock data.

    Parameters:
    - file_path (str): Path to the CSV file containing historical data with 'Date' and 'volume' columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the 'Date' and 'volume' columns.
    """
    # Load historical data from CSV
    stock_data = pd.read_csv(file_path, index_col=0)
    stock_data.index.name = "Date"  # Name the index as 'Date'
    stock_data.reset_index(inplace=True)  # Convert index to a column for merging

    # Select only the Date and volume columns
    volume_data = stock_data[["Date", "volume"]]

    return volume_data


def load_cboe_vix(file_path="data/^VIX.csv"):
    vix_data = pd.read_csv(file_path, index_col=0)
    vix_data.index.name = "Date"
    vix_data.reset_index(inplace=True)
    if "close" not in vix_data.columns:
        raise ValueError("The dataset must contain a 'close' column for VIX prices.")
    vix_data["VIX"] = np.log(vix_data["close"] / vix_data["close"].shift(1))
    vix_data.dropna(inplace=True)
    return vix_data[["Date", "VIX"]]


def load_eur_chf(file_path="data/EURCHF=X.csv"):
    eur_chf_data = pd.read_csv(file_path, index_col=0)
    eur_chf_data.index.name = "Date"
    eur_chf_data.reset_index(inplace=True)
    if "close" not in eur_chf_data.columns:
        raise ValueError(
            "The dataset must contain a 'close' column for EUR/CHF prices."
        )
    eur_chf_data["EURCHF"] = np.log(
        eur_chf_data["close"] / eur_chf_data["close"].shift(1)
    )
    eur_chf_data.dropna(inplace=True)
    return eur_chf_data[["Date", "EURCHF"]]


def load_oil_prices(file_path="data/CL=F.csv"):
    """
    Load historical oil prices from a CSV file and calculate log returns.

    Parameters:
        file_path (str): Path to the CSV file containing oil prices data.

    Returns:
        pd.DataFrame: A DataFrame with the 'Date' and 'log_return' for oil prices.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    oil_data = pd.read_csv(file_path, index_col=0)
    oil_data.index.name = "Date"
    oil_data.reset_index(inplace=True)

    # Calculate the daily log returns
    oil_data["oil_log_return"] = np.log(
        oil_data["adjclose"] / oil_data["adjclose"].shift(1)
    )
    oil_data.dropna(inplace=True)

    return oil_data[["Date", "oil_log_return"]]


def load_gold_prices(file_path="data/GC=F.csv"):
    """
    Load historical gold prices from a CSV file and calculate log returns.

    Parameters:
        file_path (str): Path to the CSV file containing gold prices data.

    Returns:
        pd.DataFrame: A DataFrame with the 'Date' and 'log_return' for gold prices.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    gold_data = pd.read_csv(file_path, index_col=0)
    gold_data.index.name = "Date"
    gold_data.reset_index(inplace=True)

    # Calculate the daily log returns
    gold_data["gold_log_return"] = np.log(
        gold_data["adjclose"] / gold_data["adjclose"].shift(1)
    )
    gold_data.dropna(inplace=True)

    return gold_data[["Date", "gold_log_return"]]


if __name__ == "__main__":
    training_dataset, testing_dataset = load()
    print("Training Dataset:")
    print(training_dataset.head())
    print("\nTesting Dataset:")
    print(testing_dataset.head())
    # Save the datasets to CSV files
    # training_dataset.to_csv('data_prepared/training_dataset.csv', index=False)
    # testing_dataset.to_csv('data_prepared/testing_dataset.csv', index=False)

    # print("Datasets have been saved as CSV files in the 'data_prepared' directory.")
