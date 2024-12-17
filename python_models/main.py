import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set up logging
logging.basicConfig(filename='sarima_model_errors.log', level=logging.ERROR)

# Load data
Sales_df = pd.read_csv("preprocessed_data/sales.csv", index_col=0)
Ingredients_df = pd.read_csv("preprocessed_data/ingredients.csv", index_col=0)

# Parse dates function with error logging
def parse_dates(date):
    for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            pass
    raise ValueError(f'No valid date format found for {date}')

# Apply the function to 'order_date' column
Sales_df['order_date'] = Sales_df['order_date'].apply(parse_dates)

# Prepare data for weekly aggregation
Sales_df = Sales_df[['order_date', 'pizza_name', 'quantity']]
sales_summary = Sales_df.groupby(['order_date', 'pizza_name']).sum().reset_index()

# Pivot the data to have pizza names as columns
sales_pivot = sales_summary.pivot(index='order_date', columns='pizza_name', values='quantity').fillna(0)

# Initialize a dictionary to store SARIMA models for each pizza
sarima_models = {}

# Fit the SARIMA model for each pizza type
for pizza_name in sales_pivot.columns:
    try:
        model = SARIMAX(sales_pivot[pizza_name], order=(1, 1, 0), seasonal_order=(1, 1, 0, 7))
        model_fit = model.fit(disp=False)
        sarima_models[pizza_name] = model_fit
    except Exception as e:
        logging.error(f"SARIMA model for {pizza_name} failed to fit. Error: {e}")

# Generate Sales Forecast for the Next Week
prediction_days = 7
predictions_sarima = {}

# Generate forecast for the next 7 days for each pizza type
for pizza_name, model in sarima_models.items():
    predictions_sarima[pizza_name] = model.predict(start=len(sales_pivot), end=len(sales_pivot) + prediction_days - 1)

# Convert the predictions into a DataFrame
predictions_df = pd.DataFrame(predictions_sarima)

# Generate predicted dates
predicted_dates = pd.date_range(start=sales_pivot.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='D')

# Assign predicted dates to predictions DataFrame
predictions_df.index = predicted_dates

# Ingredients DataFrame preparation
ingredients_df = Ingredients_df[['pizza_name', 'pizza_ingredients', 'Items_Qty_In_Grams']]
ingredients_df.rename(columns={'Items_Qty_In_Grams': 'items_qty'}, inplace=True)

# Create a dictionary to store ingredient quantities
ingredient_quantities = {}

# Get the ingredients for each pizza and calculate the required quantities
for pizza_name in predictions_df.columns:
    predicted_quantity = predictions_df[pizza_name].sum()
    pizza_ingredients = ingredients_df[ingredients_df['pizza_name'] == pizza_name]
    
    if pizza_ingredients.empty:
        logging.warning(f"No ingredient data found for {pizza_name}. Skipping ingredient calculation.")
        continue
    
    for index, row in pizza_ingredients.iterrows():
        ingredient = row['pizza_ingredients']
        ingredient_qty = row['items_qty']
        required_quantity = predicted_quantity * ingredient_qty
        
        # Convert required quantity from grams to kilograms
        required_quantity_kg = required_quantity / 1000  # 1 kg = 1000 grams
        
        # Add to the dictionary (store in kg)
        ingredient_quantities[ingredient] = ingredient_quantities.get(ingredient, 0) + required_quantity_kg

# Create a DataFrame from the ingredient quantities
ingredient_requirements_df = pd.DataFrame.from_dict(ingredient_quantities, orient='index', columns=['required_quantity_kg'])

# Create a purchase order DataFrame
purchase_order_df = ingredient_requirements_df.copy()

# Add a column for the unit of measure (in kg)
purchase_order_df['unit'] = 'kg'

# Rename the columns for better readability
purchase_order_df = purchase_order_df.rename(columns={'required_quantity_kg': 'quantity'})

# Print the purchase order
print("Purchase Order:")
print(purchase_order_df.to_string())

# Optionally, save the purchase order as a file
purchase_order_df.to_csv("purchase_order_kg.txt")