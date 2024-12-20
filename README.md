# Dominos Sales Forecasting and Purchase Order System

## Problem Statement:

Dominos wants to optimize the process of ordering ingredients by predicting future sales and creating a purchase order. By accurately forecasting sales, Dominos can ensure that it has the right amount of ingredients in stock, minimizing waste and preventing stockouts. This project aims to leverage historical sales data and ingredient information to develop a predictive model and generate an efficient purchase order system.

---

## Table of Contents
- [Dominos Sales Forecasting and Purchase Order System](#dominos-sales-forecasting-and-purchase-order-system)
  - [Problem Statement:](#problem-statement)
  - [Table of Contents](#table-of-contents)
  - [Business Use Cases:](#business-use-cases)
  - [Domain](#domain)
  - [Data Explanation](#data-explanation)
  - [Approach:](#approach)
    - [1. Data Preprocessing and Exploration](#1-data-preprocessing-and-exploration)
    - [2. Sales Prediction](#2-sales-prediction)
    - [3. Purchase Order Generation](#3-purchase-order-generation)
  - [Results:](#results)
  - [Project Report](#project-report)
  - [Skills Takeaway From This Project](#skills-takeaway-from-this-project)
  - [How to Setup this project](#how-to-setup-this-project)
    - [Softwares needed](#softwares-needed)
    - [Code](#code)
    - [Python packages](#python-packages)

---

## Business Use Cases:

- **Inventory Management**: Ensuring optimal stock levels to meet future demand without overstocking.

- **Cost Reduction**: Minimizing waste and reducing costs associated with expired or excess inventory.

- **Sales Forecasting**: Accurately predicting sales trends to inform business strategies and promotions.

- **Supply Chain Optimization**: Streamlining the ordering process to align with predicted sales and avoid disruptions.

---

## Domain

**Food Service Industry**

---

## Data Explanation

1. **Sales Data**: Historical sales records including date, pizza type, quantity sold, price, category, and ingredients.
2. **Ingredient Data**: Ingredient requirements for each pizza type, specifying the amount needed per pizza (pizza type, ingredient, quantity needed).
3. **Dated**: Data from 01-01-2015 to 31-12-2015
4. **Unique Values:** Contains 91 unique pizzas and 64 unique pizza ingredients

---

## Approach:
### 1. Data Preprocessing and Exploration
- **Data Cleaning**: Remove any missing or inconsistent data entries, handle outliers, and format the data appropriately.
- **Exploratory Data Analysis (EDA)**: Analyze sales trends, seasonality, and patterns in the historical sales data. Visualize the data to identify significant features.

### 2. Sales Prediction
- **Feature Engineering**: Create relevant features from the sales data, such as day of the week, month, promotional periods, and holiday effects.
- **Model Selection**: Choose an appropriate time series forecasting model (e.g., ARIMA, SARIMA, Prophet, LSTM, Regression Model).
- **Model Training**: Train the predictive model on the historical sales data.
- **Model Evaluation**: Use the metric Mean Absolute Percentage Error (MAPE) to evaluate model performance.

### 3. Purchase Order Generation
- **Sales Forecasting**: Predict pizza sales for the next one week using the trained model.
- **Ingredient Calculation**: Calculate the required quantities of each ingredient based on the predicted sales and the ingredient dataset.
- **Purchase Order Creation**: Generate a detailed purchase order listing the quantities of each ingredient needed for the forecasted sales period.

---

## Results:

- **Accurate Sales Predictions**: A reliable model that predicts pizza sales for future periods.
- **Purchase Order Creation**: A comprehensive purchase order detailing the required ingredients for the forecasted sales period.

---

## Project Report

[Slides](https://docs.google.com/presentation/d/1eFGWK_jWFkWJ2upx10ZEFvOmsPl78qXtjZDFhPosNHQ)

---

## Skills Takeaway From This Project

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Time Series Forecasting
- Predictive Modeling
- Business Decision Making

---

## How to Setup this project
### Softwares needed
1. IDE (VS Code)
2. Jupyter Notebook
3. Python
4. Git (with git bash)

---

### Code

Clone this repository and ```cd``` into that directory
``` 
git clone https://github.com/Gengatharan1/ds-dominos_purchase_order_analysis.git

cd dominos_purchase_order_analysis
```

### Python packages

Install all necessary packages
``` 
pip install -r requirements.txt
```

^ [Back to table of contents](#table-of-contents)