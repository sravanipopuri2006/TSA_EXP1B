# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 22.08.2025

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

# Load the new dataset
data = pd.read_csv('/content/Gold price - gold_data.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Aggregate data to daily frequency by taking the mean price
# This resolves the issue of duplicate dates for time series analysis
daily_data = data.groupby('Date')['Price ($)'].mean().reset_index()

# Set 'Date' column as index for the aggregated data
daily_data.set_index('Date', inplace=True)

# Define the target series for analysis (using 'Price ($)' column)
# Ensure the column name matches exactly including spaces and special characters
target_series_name = 'Price ($)'
target_series = daily_data[target_series_name]

# --- Time Series Transformations ---

# 1. Regular Differencing on the target series
# This helps remove trends and stabilize the mean
daily_data['price_diff'] = target_series - target_series.shift(1)

# 2. Seasonal Decomposition on the original target series (additive model, period=12 for yearly seasonality)
# A period of 12 is chosen assuming a monthly frequency and yearly seasonality in sales data.
# Adjust 'period' if your data has different seasonal patterns (e.g., quarterly=4, weekly=52)
result_original = seasonal_decompose(target_series, model='additive', period=12)
daily_data['price_seasonal_resid'] = result_original.resid # The residual component after decomposition

# 3. Log Transform
# Log transformation can help stabilize variance in series with increasing variability over time
daily_data['price_log'] = np.log(target_series)

# 4. Regular Differencing on the Log Transformed series
# Combining log transform with differencing can help achieve stationarity
daily_data['price_log_diff'] = daily_data['price_log'] - daily_data['price_log'].shift(1)

# 5. Seasonal Decomposition on the Log Differenced series
# .dropna() is used because differencing creates NaN values at the beginning of the series
result_log_diff = seasonal_decompose(daily_data['price_log_diff'].dropna(), model='additive', period=12)
daily_data['price_log_seasonal_resid'] = result_log_diff.resid # Residuals from this decomposition

# --- Plotting ---

# Create a figure with a suitable size for multiple subplots
# Increased figure height to accommodate 6 plots
plt.figure(figsize=(16, 18))

# Plot 1: Original Price Data
plt.subplot(6, 1, 1)
plt.plot(target_series, label='Original Price')
plt.legend(loc='best')
plt.title(f'Original {target_series_name} Data Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)


# Plot 2: Regular Differencing of Price
plt.subplot(6, 1, 2)
plt.plot(daily_data['price_diff'].dropna(), label='Regular Differencing') # dropna() for clearer plot
plt.legend(loc='best')
plt.title('Regular Differencing of Price ($)')
plt.xlabel('Date')
plt.ylabel('Differenced Price ($)')
plt.grid(True)

# Plot 3: Seasonal Residuals (from original series decomposition)
plt.subplot(6, 1, 3)
plt.plot(daily_data['price_seasonal_resid'].dropna(), label='Seasonal Residuals (Original Series)')
plt.legend(loc='best')
plt.title('Seasonal Residuals (from Original Price Decomposition)')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)

# Plot 4: Log Transformed Price
plt.subplot(6, 1, 4)
plt.plot(daily_data['price_log'].dropna(), label='Log Transformed Price')
plt.legend(loc='best')
plt.title('Log Transformed Price ($)')
plt.xlabel('Date')
plt.ylabel('Log Price ($)')
plt.grid(True)

# Plot 5: Log Transformation and Regular Differencing
plt.subplot(6, 1, 5)
plt.plot(daily_data['price_log_diff'].dropna(), label='Log Transformation and Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing of Price ($)')
plt.xlabel('Date')
plt.ylabel('RDiff(Log(Price ($)))')
plt.grid(True)

# Plot 6: Log Transformation, Regular Differencing, and Seasonal Differencing (Residuals)
plt.subplot(6, 1, 6)
plt.plot(daily_data['price_log_seasonal_resid'].dropna(), label='Log, Reg. Diff. & Seasonal Diff. Residuals')
plt.legend(loc='best')
plt.title('Log, Regular Differencing and Seasonal Differencing Residuals of Price ($)')
plt.xlabel('Date')
plt.ylabel('SDiff(RDiff(Log(Price ($))))')
plt.grid(True)

# Adjust subplot parameters for a tight layout
plt.tight_layout()
plt.show()
```


### OUTPUT:
<img width="843" height="747" alt="image" src="https://github.com/user-attachments/assets/0e0b3ef6-fd3f-43fc-a36a-ef0bc2e53886" />
<img width="852" height="168" alt="image" src="https://github.com/user-attachments/assets/03bc31e9-5265-46ae-9c78-309294140441" />





### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
