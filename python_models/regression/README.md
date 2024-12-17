### **REGRESSION MODEL**

In time-series analysis, a **regression model** is used to predict future values based on historical data. It helps identify relationships between a dependent variable (target) and one or more independent variables (predictors or features), which may include time-based features like year, month, week, or day. 

### **Purpose and Uses:**
- **Forecasting:** Predict future values based on past data.
- **Trend and Seasonality Detection:** Capture trends or seasonal patterns using time-related features.
- **Feature Relationships:** Understand how time-related features (e.g., month, day of the week) influence the target variable.

### **Types of Regression Models:**
1. **Linear Regression**: Used when the relationship between the dependent variable and predictors is linear.
2. **Multiple Linear Regression**: Extends linear regression by using multiple predictors (e.g., time-based features).
3. **Polynomial Regression**: For nonlinear relationships between variables.
4. **Ridge and Lasso Regression**: Regularized versions of linear regression to prevent overfitting.
5. **Logistic Regression**: Used when the target is binary (not as common in time-series but can be used for specific cases like event prediction).

### **Parameters of a Regression Model:**
- **Intercept (β₀)**: The predicted value when all independent variables are 0. It represents the baseline value of the dependent variable.
- **Coefficients (β₁, β₂, ...)**: The slopes associated with each predictor variable, showing how much the dependent variable changes with a one-unit change in the predictor.
- **R-squared (R²)**: Measures the proportion of variance in the dependent variable that is explained by the independent variables.
- **P-value**: Tests the hypothesis that each coefficient is different from zero. A lower p-value indicates a stronger predictor.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, representing the error in the same units as the dependent variable.
- **Mean Absolute Percentage Error (MAPE)**: The average absolute percentage difference between predicted and actual values.
- **Adjusted R-squared**: An adjusted version of R² that accounts for the number of predictors in the model, penalizing for overfitting.

### **Steps to Apply Regression Model:**
1. **Data Preparation:** Clean and preprocess the data, and aggregate it (e.g., converting daily data to weekly/monthly).
2. **Feature Engineering:** Create time-based features like year, month, day, week of the year, etc.
3. **Model Training:** Fit the regression model using the training data.
4. **Model Evaluation:** Use metrics like RMSE, MAPE, or R² to evaluate the model’s performance.
5. **Forecasting:** Use the trained model to predict future values.

### **Key Considerations:**
- **Stationarity:** Time-series data may need to be stationary (constant mean and variance) for regression models to work effectively.
- **Autocorrelation:** Consider lagged features (previous time values) to account for temporal dependencies.
- **Seasonality and Trends:** Ensure that the model captures these components using time-based features.