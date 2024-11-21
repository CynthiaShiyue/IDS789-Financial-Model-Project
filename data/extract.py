from typing import Union
import datetime
from yahoo_fin import stock_info as si
import os


def extract_histData(Name: str, startdate: Union[str, datetime.date], enddate: Union[str, datetime.date]) -> str:
    """
    Extract historical data from Yahoo Finance and save it as a CSV in the data directory.

    Parameters:
    - Name (str): The stock ticker symbol (e.g., "UBS" for UBS Group AG).
    - startdate (str or datetime.date): The start date for historical data extraction.
    - enddate (str or datetime.date): The end date for historical data extraction.

    Returns:
    - str: The filename of the saved CSV file.
    """
    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)
    
    # Retrieve historical data
    hist_data = si.get_data(Name, start_date=startdate, end_date=enddate)
    
    # Save the data to a CSV file in the 'data' directory
    filename = f"data/{Name}.csv"
    hist_data.to_csv(filename, index=True)
    print(f"Data saved to {filename}")
    return filename

if __name__ == "__main__":
    # Example usage
    extract_histData("UBS", "2021-01-04", "2024-01-05")
    
    # A: SPY500 & FTSE 100 index
    
    # B: ^VIX & EUR/CHF

    # C: Oil Prices & Gold Price
    
    # D: two Competitors
    extract_histData("MS", "2021-01-04", "2024-01-05")
    extract_histData("DB", "2021-01-04", "2024-01-05")


