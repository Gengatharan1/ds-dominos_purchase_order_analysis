# **ARIMA MODEL**

## **IMPORTANT PARAMETERS NEEDS TUNING TO GET BEST MODEL**

**p (AutoRegressive term):** The number of past observations (lags) used to predict the future value.

**d (Differencing term):** The number of times the data is differenced to make it stationary.

**q (Moving Average term):** The number of past forecast errors used to improve future predictions.

## Steps to implemented in ARIMA Model:

### **STEP 1: Prepare Weekly Sales Data**
- Convert `order_date` to datetime format.
- Group sales data by week (using the start time of each week) and sum up the quantity for each week.

### **STEP 2: Train-Test Split**
- Split the weekly sales data into training and testing datasets (80% train, 20% test).

### **STEP 3: Define Evaluation Metrics**
- **MAPE (Mean Absolute Percentage Error)**: Measures the accuracy of predictions as a percentage.
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors.

### **STEP 4: Perform Stationarity Check**
- Apply the **Augmented Dickey-Fuller (ADF)** test to check if the data is stationary.
- Print the ADF statistic and p-value.
  - If p-value ≤ 0.05, the data is stationary.
  - If p-value > 0.05, the data is non-stationary and may require differencing.

### **STEP 5: ARIMA Model Tuning**
- Use grid search to tune the ARIMA model by testing different combinations of `p`, `d`, and `q` parameters.
- **Parallel processing** is used to speed up the search for the best ARIMA model.

### **STEP 6: Set Parameters and Tune ARIMA Model**
- Define the range of values for the ARIMA parameters (`p`, `d`, `q`).
- Find the best combination of parameters using the grid search approach.

### **STEP 7: Format and Display Predictions**
- Convert the predictions to a pandas Series and display them alongside actual sales data.

### **STEP 8: Plot Actual vs Predicted Sales**
- Plot the actual vs predicted sales data on a graph to visually compare the performance of the ARIMA model.

