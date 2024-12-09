import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] <= 0.05:
        print("Stationary.")
    else:
        print("Non-Stationary.")


def calculate_vif(df):
    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


def check_missing_data(df):
    # Check for missing values
    print(df.isnull().sum())

    # Fill missing values (example using mean imputation)
    #df = df.fillna(df.mean())
    return "check missing"


def check_correlation(df):
    correlation_matrix = df.corr()
    print(correlation_matrix["UBS_Stock_Return"])
    return "check correlation"

# if __name__ == "__main__":
# df = pd.DataFrame(data)

# # Step 1: Check Stationarity
# check_stationarity(df["UBS_Stock_Return"])

# # Step 2: Check Missing Data
# check_missing_data(df)

# # Step 3: Check Correlations
# check_correlation(df)

# # Step 4: Calculate VIF
# vif_result = calculate_vif(df)
# print(vif_result)