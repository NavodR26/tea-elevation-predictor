import pandas as pd
import numpy as np
from datetime import datetime

# Create test data matching your tea auction format
np.random.seed(42)

# Generate realistic tea auction data
years = list(range(2018, 2026))
elevations = ['High Grown', 'Medium Grown', 'Low Grown', 'Uva', 'Nuwara Eliya', 'Dimbula']
data = []

for year in years:
    # 50 sales per year (typical for tea auctions)
    for sale_no in range(1, 51):
        for elevation in elevations:
            # Realistic price ranges for different elevations
            base_prices = {
                'High Grown': 450,
                'Medium Grown': 380,
                'Low Grown': 320,
                'Uva': 480,
                'Nuwara Eliya': 520,
                'Dimbula': 470
            }
            
            # Add seasonal variation and trends
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * sale_no / 50)
            trend_factor = 1 + 0.02 * (year - 2018)
            noise = np.random.normal(0, 0.1)
            
            price = base_prices[elevation] * seasonal_factor * trend_factor * (1 + noise)
            quantity = np.random.uniform(5000, 25000)  # kg
            
            data.append({
                'Year': year,
                'Sale Number': sale_no,
                'Elevation': elevation,
                'Quantity': round(quantity, 1),
                'Average Price': round(price, 2)
            })

# Create DataFrame and save as Excel
df = pd.DataFrame(data)
print(f"Created test data with {len(df)} records")
print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
print(f"Elevations: {df['Elevation'].unique()}")
print(f"Price range: {df['Average Price'].min():.2f} - {df['Average Price'].max():.2f}")

# Save as Excel file
df.to_excel('test_tea_data.xlsx', index=False)
print("Test data saved as test_tea_data.xlsx")