import pandas as pd
import numpy as np
import os
from functools import reduce

def load():
    #Cynthia
    ubs_stock_return=load_stock_return(file_path="data/UBS.csv")
    ubs_Bid_Ask_Spread_df=load_bid_ask_spread("data/UBS.csv")
    ubs_volume_df=trading_volume("data/UBS.csv")
    
    ubs_stock_return.reset_index(inplace=True)
    ubs_Bid_Ask_Spread_df.reset_index(inplace=True)
    ubs_volume_df.reset_index(inplace=True)
   
    # Kaisen's data
    vix_data = load_cboe_vix(file_path="data/^VIX.csv")
    eur_chf_data = load_eur_chf(file_path="data/EURCHF=X.csv")

    # Combine all datasets
    dfs = [
        ubs_stock_return,
        ubs_Bid_Ask_Spread_df,
        ubs_volume_df,
        vix_data,
        eur_chf_data,
    ]
    # Merge datasets, avoiding duplicate column conflicts
    df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)

    
    # split into training dataset(R),and testing dataset(P) 
    df["Date"] = pd.to_datetime(df["Date"]) # Convert the "Date" column to datetime format if it is not already
    training_dataset = df[df["Date"] < "2023-01-02"]
    testing_dataset = df[df["Date"] >= "2023-01-02"]
    return training_dataset,testing_dataset


def load_stock_return(file_path="data/UBS.csv"):
    """
    Load stock data from a CSV file and calculate stock returns.

    Parameters:
        file_path (str): Path to the CSV file containing stock data.

    Returns:
        pd.DataFrame: A DataFrame with the 'Date' and log returns.
    """
    # Load the stock data, assuming the first column is the index (e.g., date)
    stock_data = pd.read_csv(file_path, index_col=0)
    
    # Check if the index is named "Date"
    if stock_data.index.name != "Date":
        stock_data.index.name = "Date"
    
    # Reset index to make 'Date' a column
    stock_data.reset_index(inplace=True)
    
    # Ensure the 'close' column is present
    if 'close' not in stock_data.columns:
        raise ValueError("The dataset must contain a 'close' column for stock prices.")
    
    # Calculate the daily log stock returns
    stock_data['log_return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))
    
    # Drop rows with NaN values resulting from the calculation
    stock_data.dropna(inplace=True)
    
    return stock_data[['Date', 'log_return']]




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
    stock_data.index.name = 'Date'  # Name the index as 'Date'
    stock_data.reset_index(inplace=True)  # Convert index to a column for merging

    
    # Use lowercase column names to match the CSV
    stock_data['high_shift'] = stock_data['high'].shift(1)
    stock_data['low_shift'] = stock_data['low'].shift(1)
    
    # Calculate beta and gamma
    stock_data['beta'] = (np.log(stock_data['high'] * stock_data['low_shift'] / (stock_data['low'] * stock_data['high_shift']))) ** 2
    stock_data['gamma'] = (np.log(stock_data['high'] / stock_data['low']) ** 2 + np.log(stock_data['high_shift'] / stock_data['low_shift']) ** 2) / 2
    
    # Calculate alpha and spread
    stock_data['alpha'] = stock_data['beta'] / stock_data['gamma']
    stock_data['Spread'] = 2 * (np.sqrt(np.exp(stock_data['alpha']) - 1))
    
    # Return only the 'Date' and 'Spread' columns
    result = stock_data[['Date', 'Spread']]

    
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
    stock_data.index.name = 'Date'  # Name the index as 'Date'
    stock_data.reset_index(inplace=True)  # Convert index to a column for merging

    
    # Select only the Date and volume columns
    volume_data = stock_data[['Date', 'volume']]
    
    return volume_data
    
    
    
    
# A

# B

# C


def load_cboe_vix(file_path="data/^VIX.csv"):
    vix_data = pd.read_csv(file_path, index_col=0)
    vix_data.index.name = "Date"
    vix_data.reset_index(inplace=True)
    if "close" not in vix_data.columns:
        raise ValueError("The dataset must contain a 'close' column for VIX prices.")
    vix_data["log_return"] = np.log(vix_data["close"] / vix_data["close"].shift(1))
    vix_data.dropna(inplace=True)
    return vix_data[["Date", "log_return"]]


def load_eur_chf(file_path="data/EURCHF=X.csv"):
    eur_chf_data = pd.read_csv(file_path, index_col=0)
    eur_chf_data.index.name = "Date"
    eur_chf_data.reset_index(inplace=True)
    if "close" not in eur_chf_data.columns:
        raise ValueError(
            "The dataset must contain a 'close' column for EUR/CHF prices."
        )
    eur_chf_data["log_return"] = np.log(
        eur_chf_data["close"] / eur_chf_data["close"].shift(1)
    )
    eur_chf_data.dropna(inplace=True)
    return eur_chf_data[["Date", "log_return"]]


if __name__ == "__main__":
    training_dataset, testing_dataset = load()
    print("Training Dataset:")
    print(training_dataset.head())
    print("\nTesting Dataset:")
    print(testing_dataset.head())

# D 


if __name__ == "__main__":
    load()
    