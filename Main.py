import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
import numpy as np

#Definition of classic cointegrated equity pairs
assets = ['XLE', 'VDE']
start_date = datetime.datetime(2021, 1, 1)
end_date = datetime.datetime(2026, 1, 1)

#Retrieve adjusted prices of the selected assets from Yahoo Finance
data = yf.download(assets, start_date, end_date, auto_adjust=True)

#Select the 'Close' column from prices
close_data = data['Close']

#Remove rows with missing values from the table completely
close_data.dropna(inplace=True)

#Plot historical price series (line graph)
close_data.plot(figsize=(20,10),title='Historical Adjusted Close Prices: State Street Energy Select Sector (XLE) vs Vanguard Energy ETF (VDE)')
plt.ylabel('Close Prices in USD $')
plt.grid(True, color='white', linestyle='--', alpha=0.5)
plt.show()


#Separating two price vectors:
series_1 = close_data['XLE']
series_2 = close_data['VDE']

#Evaluate their cointegration based on p value:
cointegration_test = coint(series_1, series_2).pvalue
if cointegration_test <0.05:
    print('Cointegrated')
else:
    print('Not Cointegrated')

#Calculating the hedge ratio (B = Beta x A + Alpha)
#1. Calculate dependent and independent variables: B = XLE; A=VDE
B = series_1
A = series_2
#2 Add constant (Alpha) to A
A_with_constant = add_constant(A)
#3. Fit an OLS Regression algorithm
Ordinary_least_squares = OLS(B, A_with_constant).fit()
#4. Extract the coefficient (Hedge Ratio) and save it in a variable
beta = Ordinary_least_squares.params['VDE']
#5. Construct the spread (it uses constant at its average level)
#Beta scales with price but the intercept adds a fixed dollar amount
spread = B - (beta * series_2)
print(f"Hedge Ratio (Beta): {beta:.4f}")
#Display the spread mean
print(f"Mean spread: {spread.mean():.6f}")
#Display the spread plot
spread.plot(figsize=(20,10), xlabel='Date', ylabel='Spread (USD)', title='Spread Oscillation over time (2021 - 2026)')
plt.show()

#Rolling Z score (How many standard deviations is the last spread from the mean spread)
rolling_mean = spread.rolling(window=20).mean()
rolling_std =spread.rolling(window=20).std()
rolling_z_score = (spread-rolling_mean)/rolling_std
rolling_z_score_without_NaaNs = rolling_z_score.dropna()
rolling_z_score_without_NaaNs.plot(figsize=(20,10), title='Rolling Z-Score Graph for spread over time (2021 - 2026)', color='green')
plt.grid(True, color='grey', linestyle='--', alpha=0.5)
plt.show()

#Discrete Trading Signals or positions
conditions = [rolling_z_score_without_NaaNs > 2.0,rolling_z_score_without_NaaNs < -2.0]
choices = [-1, 1]
signals = np.select(conditions, choices, default=0)
print(signals)
#Creating a series with the signals and dates:
signals_series = pd.Series(signals, index=rolling_z_score_without_NaaNs.index)
print(signals_series)
positions = signals_series.ffill()
print(positions)
