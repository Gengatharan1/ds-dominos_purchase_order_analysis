# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ARIMA MODEL
import itertools
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from joblib import Parallel, delayed
from tqdm import tqdm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error as mape

import warnings
warnings.filterwarnings("ignore")   # Suppressing the warnings

# Load merged data
merged_data = pd.read_csv("data/merged_data.csv")
merged_data.head()

# STEP 1: Prepare Weekly Sales Data
def prepare_weekly_sales(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    weekly_sales = df.groupby(df['order_date'].dt.to_period('W').apply(lambda r: r.start_time))['quantity'].sum()
    return weekly_sales

# Load merged data
pizza_sales_weekly = prepare_weekly_sales(merged_data)

# STEP 2: Train-Test Split
train_size = int(0.8 * len(pizza_sales_weekly))
train, test = pizza_sales_weekly[:train_size], pizza_sales_weekly[train_size:]

# Display Train-Test Split
print("Train Data:\n", train.tail())
print("\nTest Data:\n", test.tail())

# STEP 3: Define Evaluation Metrics
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# STEP 4: Perform Stationarity Check
def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is not stationary; differencing may be required.")

print("\nStationarity Check for Training Data:")
check_stationarity(train)

# STEP 5: ARIMA Model Tuning
def tune_arima_model(train, test, p_values, d_values, q_values):
    best_score, best_params, best_predictions = float("inf"), None, None

    # Grid search for ARIMA parameters
    def evaluate_arima_params(p, d, q):
        try:
            model = ARIMA(train, order=(p, d, q)).fit()
            predictions = model.forecast(steps=len(test))
            score = mape(test, predictions)
            return (score, (p, d, q), predictions)
        except Exception as e:
            print(f"Error for parameters (p={p}, d={d}, q={q}): {e}")
            return None

    results = Parallel(n_jobs=-1)(delayed(evaluate_arima_params)(p, d, q) for p, d, q in itertools.product(p_values, d_values, q_values))

    # Filter out failed models and find the best one
    results = [res for res in results if res is not None]
    for score, params, predictions in results:
        if score < best_score:
            best_score, best_params, best_predictions = score, params, predictions

    print(f"\nBest ARIMA Model MAPE: {best_score:.4f}\nBest Parameters: {best_params}")
    return best_predictions, best_score, best_params

# STEP 6: Set Parameters and Tune ARIMA Model
p_values, d_values, q_values = range(0, 3), range(0, 2), range(0, 3)
arima_predictions, arima_mape_score, best_params = tune_arima_model(train, test, p_values, d_values, q_values)

# STEP 7: Format and Display Predictions
arima_predictions = pd.Series(arima_predictions, index=test.index)
print("\nARIMA Predictions:\n", arima_predictions)

# STEP 8: Plot Actual vs Predicted Sales
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual Sales', color='blue', marker='o')
plt.plot(arima_predictions.index, arima_predictions, label='Predicted Sales', color='red', linestyle='--', marker='x')
plt.title('ARIMA Model : Predictions vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()