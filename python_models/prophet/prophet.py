# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")   # Suppressing the warnings

# PROPHET MODEL
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error as mape

# Load merged data
merged_data = pd.read_csv("data/merged_data.csv")
merged_data.head()

# STEP 1: Prepare Data for Prophet
# Rename columns to match Prophet's expected format
df = merged_data.rename(columns={'order_date': 'ds', 'quantity': 'y'})  # Assign variable name as ds (train_feature)and y (target))
df = df[['ds', 'y']]

# STEP 2: Pizza sales by week
def prepare_weekly_sales_for_prophet(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    weekly_sales = df.groupby(df['order_date'].dt.to_period('W').apply(lambda r: r.start_time))['quantity'].sum().reset_index()
    weekly_sales.columns = ['ds', 'y']  # Prophet requires columns 'ds' for date and 'y' for the target variable
    return weekly_sales
pizza_sales_weekly = prepare_weekly_sales_for_prophet(merged_data)

# STEP 3: Train Test Split
train_size = int(0.8 * len(pizza_sales_weekly))
train, test = pizza_sales_weekly[:train_size], pizza_sales_weekly[train_size:]

# STEP 4: Calculate MAPE
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual))

# STEP 5: Initialize and train the model
def best_prophet_model(train, test):
    model = Prophet()
    model.fit(train)

    # Create a DataFrame with future dates for predictions
    future = model.make_future_dataframe(periods=len(test), freq='W')

    # Generate predictions for future dates
    forecast = model.predict(future)
    
    # Extract the predictions for the test period
    predictions = forecast['yhat'][-len(test):].values
    
    # Calculate MAPE
    prophet_mape = mape(test['y'].values, predictions)
    print(f"\nBest Prophet Model MAPE: {prophet_mape:.4f}")
    return predictions, prophet_mape

# Train and evaluate the Prophet model
prophet_predictions, prophet_mape_score = best_prophet_model(train, test)

# Formating the predictions for display
prophet_predictions = pd.Series(prophet_predictions, index=test['ds'])

print("\nPredictions:")
print(prophet_predictions)

# STEP 6:  Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test['ds'], test['y'], label='Actual Sales', color='blue', marker='o')
plt.plot(prophet_predictions.index, prophet_predictions, label='Predicted Sales', color='red', linestyle='--', marker='x')
plt.title('Prophet Model : Predictions vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()