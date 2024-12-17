# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SARIMA MODEL
import itertools
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from joblib import Parallel, delayed
from tqdm import tqdm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse, r2_score

import warnings
warnings.filterwarnings("ignore")   # Suppressing the warnings

# Load merged data
merged_data = pd.read_csv("data/merged_data.csv")

# STEP 1: Pizza sales by week
def prepare_weekly_sales(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    # Check for missing values and handle them
    df.dropna(subset=['order_date', 'quantity'], inplace=True)
    weekly_sales = df.groupby(df['order_date'].dt.to_period('W').apply(lambda r: r.start_time))['quantity'].sum()
    return weekly_sales

pizza_sales_weekly = prepare_weekly_sales(merged_data)

# STEP 2: Train-test split
train_size = int(0.8 * len(pizza_sales_weekly))
train, test = pizza_sales_weekly[:train_size], pizza_sales_weekly[train_size:]

# STEP 3: Evaluate MAPE
def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / (actual)))

# STEP 4: SARIMA Model Training and Hyperparameter Tuning
def best_sarima_model(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False).fit(disp=False)
        predictions = model.forecast(steps=len(test))
        sarima_mape = mape(test, predictions)
        # print(f"\nBest SARIMA Model MAPE: {sarima_mape:.4f}")
        return predictions, sarima_mape, model
    except Exception as e:
        print(f"Error during SARIMA model training: {e}")
        return None, None, None

# STEP 5: Hyperparameter Tuning using Grid Search
def grid_search_sarima(train, test, seasonal_period=7):
    p = d = q = range(0, 3)  # Example range for p, d, q
    P = D = Q = range(0, 2)  # Example range for seasonal P, D, Q
    seasonal_order = [seasonal_period]
    param_grid = list(itertools.product(p, d, q, P, D, Q, seasonal_order))

    best_mape = float('inf')
    best_model = None
    best_params = None

    for param in param_grid:
        try:
            model = SARIMAX(train, order=(param[0], param[1], param[2]), seasonal_order=(param[3], param[4], param[5], param[6]))
            result = model.fit(disp=False)
            predictions = result.forecast(steps=len(test))
            sarima_mape = mape(test, predictions)
            if sarima_mape < best_mape:
                best_mape = sarima_mape
                best_model = result
                best_params = param
        except Exception as e:
            continue  # Skip models that fail to train
    
    print(f"Best SARIMA parameters: {best_params}")
    print(f"Best MAPE: {best_mape:.4f}")
    return best_model

# Run Grid Search for Best SARIMA Model
best_sarima_model_result = grid_search_sarima(train, test)

# STEP 6: Train, Evaluate the SARIMA model
sarima_predictions, sarima_mape_score, sarima_model = best_sarima_model(train, test)

if sarima_predictions is not None:
    sarima_predictions = pd.Series(sarima_predictions, index=test.index)
    print("\nPredictions:")
    print(sarima_predictions)

    # STEP 7: Plotting Actual vs Predicted Values
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test.values, label='Actual Sales', color='blue', marker='o')
    plt.plot(sarima_predictions.index, sarima_predictions, label='Predicted Sales', color='red', linestyle='--', marker='x')
    plt.title('SARIMA Model : Predictions vs Actual Sales')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # STEP 8: Residual Diagnostics
    residuals = sarima_model.resid
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, label="Residuals", color='purple')
    plt.axhline(0, linestyle='--', color='black', linewidth=1)
    plt.title('SARIMA Residuals')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ACF and PACF of Residuals to check for autocorrelation
    plot_acf(residuals)
    plot_pacf(residuals)
    plt.show()
else:
    print("SARIMA model failed. Please check the data or parameters.")