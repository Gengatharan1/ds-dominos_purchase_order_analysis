import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# REGRESSION MODEL
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_squared_error as mse, r2_score

# Load merged data
merged_data = pd.read_csv("data/merged_data.csv")
merged_data.head()

# STEP 1: Aggregate Weekly Sales
def prepare_weekly_sales(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    weekly_sales = df.groupby(df['order_date'].dt.to_period('W').apply(lambda r: r.start_time))['quantity'].sum().reset_index()
    weekly_sales.rename(columns={'order_date': 'start_of_week'}, inplace=True)
    return weekly_sales

# STEP 2: Feature Engineering
def create_regression_features(df):
    df['week_of_year'] = df['start_of_week'].dt.isocalendar().week
    df['day_of_week'] = df['start_of_week'].dt.weekday
    df['month'] = df['start_of_week'].dt.month
    df['year'] = df['start_of_week'].dt.year
    return df

# MAPE Function
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# STEP 3: Prepare Weekly Sales and Feature Engineering
pizza_sales_weekly = prepare_weekly_sales(merged_data)
pizza_sales_weekly = create_regression_features(pizza_sales_weekly)

# STEP 4: Train-Test Split
train_size = int(0.8 * len(pizza_sales_weekly))
train, test = pizza_sales_weekly[:train_size], pizza_sales_weekly[train_size:]

X_train = train[['week_of_year', 'day_of_week', 'month', 'year']]
y_train = train['quantity']
X_test = test[['week_of_year', 'day_of_week', 'month', 'year']]
y_test = test['quantity']

# STEP 5: Define Regression Model and Evaluate
def best_regression_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    regression_mape = mape(y_test, predictions)
    print(f"Best Regression Model MAPE: {regression_mape:.4f}")
    return predictions, regression_mape

# Evaluate Regression Model
regression_predictions, regression_mape_score = best_regression_model(X_train, y_train, X_test, y_test)

# Format Predictions
regression_predictions = pd.Series(regression_predictions, index=test['start_of_week'])

print("\nPredictions:")
print(regression_predictions)

# STEP 6: Plot Actual vs Predicted Sales
plt.figure(figsize=(12, 6))
plt.plot(test['start_of_week'], y_test, label='Actual Sales', color='blue', marker='o')
plt.plot(regression_predictions.index, regression_predictions, label='Predicted Sales', color='red', linestyle='--', marker='x')
plt.title('Regression Model: Predictions vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid()