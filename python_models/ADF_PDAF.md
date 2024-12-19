### Dickey-Fuller test

In statistics, the Dickey–Fuller test tests the null hypothesis that a unit root is present in an autoregressive (AR) time series model.

#### Augmented Dickey-Fuller (ADF) test
This is an extended version of the Dickey-Fuller test that can be used for more complicated time series models. The ADF test statistic is a negative number, and the more negative it is, the stronger the rejection of the unit root hypothesis. 

#### Critical value
The critical value for the test is found in the Dickey-Fuller table. 

#### Decision
If the test statistic is less than the critical value, or the p-value is less than a specified significance level, the null hypothesis is rejected. This means the time series is considered stationary. 

#### Unit root
A unit root occurs when |φ| = 1, which means the time series is not stationary. When φ = 1, the process is a random walk without drift. 

#### Akaike information criterion (AIC)
This is an estimator of prediction error and thereby relative quality of statistical models for a given set of data. In estimating the amount of information lost by a model, AIC deals with the trade-off between the goodness of fit of the model and the simplicity of the model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.

### Autocorrelation (ACF) vs Partial Autocorrelation (PACF)

Both **autocorrelation** and **partial autocorrelation** are statistical tools used to analyze the relationship between time series observations at different lags. However, they differ in what they measure and how they are interpreted.

---

### 1. **Autocorrelation (ACF)**

#### **Definition:**
Autocorrelation measures the correlation between a time series and its lagged version over various time steps (lags).

#### **Key Points:**
- It considers the total correlation of a series with its lagged values.
- Each lag's autocorrelation includes the influence of intermediate lags. For example, the correlation at lag k includes the effects of lags 1, 2, ..., k-1.
- Values range between -1 and +1, where:
  - +1 : Perfect positive correlation.
  - -1 : Perfect negative correlation.
  -  0 : No correlation.

#### **Use Case:**
- To identify patterns or periodicities in a time series.
- To check if past values have a strong relationship with current values.

---

### 2. **Partial Autocorrelation (PACF)**

#### **Definition:**
Partial autocorrelation measures the correlation between a time series and its lagged values after removing the effects of intermediate lags.

#### **Key Points:**
- It isolates the direct relationship between a time series and its lag k, excluding the influence of all shorter lags ( 1, 2, ..., k-1).
- Helps understand the "pure" correlation at each lag.

#### **Use Case:**
- To determine the order of the **AR** (Auto-Regressive) part in an ARIMA model by identifying where the PACF plot cuts off.

### **Example:**
- If you have a time series of daily temperatures:
  - The **ACF** for lag 3 tells you how today’s temperature correlates with the temperature three days ago, considering all intermediate days.
  - The **PACF** for lag 3 tells you the direct relationship between today’s temperature and the temperature three days ago, excluding the effect of days 1 and 2.

---

### Visual Representation
- **ACF Plot:** Often shows a slow decay if the series has strong serial correlation.
- **PACF Plot:** Shows sharp spikes at significant lags and cuts off for others.

Understanding both ACF and PACF is crucial for model selection and diagnosing time series properties effectively.