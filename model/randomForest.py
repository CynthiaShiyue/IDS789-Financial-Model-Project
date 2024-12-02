from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from sklearn.model_selection import GridSearchCV
# Load training and test datasets
train = pd.read_csv(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\training_dataset.csv"
)
test = pd.read_csv(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\data_prepared\testing_dataset.csv"
)

# Ensure Date is in datetime format
train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

# target-explanatory variables split
X_train = train.drop(columns=["UBS log_return", "Date"])
y_train = train["UBS log_return"]
X_test = test.drop(columns=["UBS log_return", "Date"])
y_test = test["UBS log_return"]

# Initialize Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Train and evaluate
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))
print("Random Forest R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))

# Plot Actual values vs Predicted values
plt.plot(test["Date"], y_test, label="Actual", color="blue", alpha=0.6)
plt.plot(test["Date"], y_pred, label="Predicted", color="red", alpha=0.6)

# Format x-axis for dates (using DateFormatter for readability)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Add labels and title
plt.title("Actual vs Predicted Stock Returns for UBS using Random Forest", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("UBS Log Return", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(
    r"C:\Users\DELL\OneDrive\Desktop\IDS789-Financial-Model-Project\model\randomforest.png"
)
plt.show()


# rf = RandomForestRegressor()

# # Set up hyperparameters to tune
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }

# # GridSearchCV to find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters
# print(f"Best Hyperparameters: {grid_search.best_params_}")

# # Use the best estimator to predict
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test)