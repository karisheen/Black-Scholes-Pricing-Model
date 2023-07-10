# Create a calculator using the black scholes model to determine the fair price for a call or put option 
# based on six variables such as volatality, type of option, underlying stock price, time, strike, and risk-free rate.

from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

# Define the variables
# S = underlying asset price
# K = strike price
# T = time to maturity
# r = risk-free rate
# sigma = volatility

# Define the d1 and d2 functions

def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

# Define the Black Scholes Call Option formula
def bs_call(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

# Define the Black Scholes Put Option formula
def bs_put(S, K, T, r, sigma):
    return K * exp(-r * T) - S + bs_call(S, K, T, r, sigma)

# Ask the user if the want to use real stock data or manually input the variables
data_choice = input ("Do you want to use real stock data or manually input the variables? (r/m): ")

if data_choice == 'r':
    # Fetch real stock data and calculate options prices
    stock = str(input("select the stock you want: "))
    data = yf.download(stock, period="1y")  # Download 1 year of data using yfinance
    current_price = round(data["Adj Close"].iloc[-1], 2)
    print("The current price of", stock, " is: ", current_price)
    choice = input("Want to price a call or a put ? (c/p): ")
    expiry = str(input("Select the expiry date (format mm-dd-YYYY): "))
    strike_price = int(input("Select the strike price: "))
    data["returns"] = data["Close"].pct_change()
    sigma = np.sqrt(252) * data["returns"].std()

    # Assume a risk-free rate of 2% or any other rate // did this beacuse yfinance was struggling to calcluate the risk free rate from the treasury data. Will fix later
    ty10y = 0.02
    t = (datetime.strptime(expiry, "%m-%d-%Y") - datetime.now()).days / 365

    if choice == "c":
        print("The Call Option Price is: ", bs_call(current_price, strike_price, t, ty10y, sigma))
    if choice == "p":
        print("The Put Option Price is: ", bs_put(current_price, strike_price, t, ty10y, sigma))

elif data_choice == 'm':
    # Ask the user to input their own values for the variables
    S = float(input("Enter the underlying price of the stock: "))  # Current price of the underlying asset
    K = float(input("Enter the strike price of the option: "))  # Strike price of the option
    T = float(input("Enter the time to expiry in days: ")) / 365  # Time to expiry in years
    r = float(input("Enter the risk-free rate as a percentage: ")) / 100  # Risk-free interest rate as a percentage
    sigma = float(input("Enter the volatility of the underlying asset as a percentage: ")) / 100  # Volatility as a percentage

    # Ask user whether they want to calculate a call or a put option
    choice = input("Want to price a call or a put ? (c/p): ")
    
    if choice == "c":
        print("The Call Price is: ", bs_call(S, K, T, r, sigma))
    if choice == "p":
        print("The Put Price is: ", bs_put(S, K, T, r, sigma))


