import pandas as pd
import numpy as np

def load():
    #Cynthia
    ubs_Bid_Ask_Spread_df=load_bid_ask_spread("/data/UBS.csv")
    ubs_volume_df=trading_volume("/data/UBS.csv")
    # A

    # B

    # C

    # D 
    
    # prepared model data
    df = pd.merge(ubs_Bid_Ask_Spread_df, ubs_volume_df, on="Date", how="inner")
    
    # split into training dataset(R),and testing dataset(P) 
    df["Date"] = pd.to_datetime(df["Date"]) # Convert the "Date" column to datetime format if it is not already
    training_dataset = df[df["Date"] < "2023-01-02"]
    testing_dataset = df[df["Date"] >= "2023-01-02"]
    return training_dataset,testing_dataset

def load_bid_ask_spread(file_path="/data/UBS.csv"):
    """
    Calculate estimated spread from historical high and low prices in a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing historical data with 'Date', 'High', and 'Low' columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the 'Date' and calculated 'Spread'.
    """
    # Load historical data from CSV
    stock_data = pd.read_csv(file_path)
    
    # Shift high and low columns to calculate beta and gamma
    stock_data['High_shift'] = stock_data['High'].shift(1)
    stock_data['Low_shift'] = stock_data['Low'].shift(1)
    
    # Calculate beta and gamma
    stock_data['beta'] = (np.log(stock_data['High'] * stock_data['Low_shift'] / (stock_data['Low'] * stock_data['High_shift']))) ** 2
    stock_data['gamma'] = (np.log(stock_data['High'] / stock_data['Low']) ** 2 + np.log(stock_data['High_shift'] / stock_data['Low_shift']) ** 2) / 2
    
    # Calculate alpha and spread
    stock_data['alpha'] = stock_data['beta'] / stock_data['gamma']
    stock_data['Spread'] = 2 * (np.sqrt(np.exp(stock_data['alpha']) - 1))
    
    result = stock_data[['Date', 'Spread']]
    
    return result

def trading_volume(file_path="/data/UBS.csv"):
    """
    Extract trading volume from historical stock data.

    Parameters:
    - file_path (str): Path to the CSV file containing historical data with 'Date' and 'volume' columns.

    Returns:
    - pd.DataFrame: A DataFrame containing the 'Date' and 'volume' columns.
    """
    # Load historical data from CSV
    stock_data = pd.read_csv(file_path)
    
    # Select only the Date and volume columns
    volume_data = stock_data[['Date', 'volume']]
    
    return volume_data
    
    
    
    
# A

# B

# C

# D 


if __name__ == "__main__":
    load()