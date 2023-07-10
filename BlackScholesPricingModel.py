# Create a calculator using the black scholes model to determine the fair price for a call or put option 
# based on six variables such as volatality, type of option, underlying stock price, time, strike, and risk-free rate.

from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web

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

if data_choice == "r":
    # Fetch real stock data and calculate option prices
    stock = str(input("Enter the stock ticker: "))
    current_price = round(web.DataReader(stock, "yahoo")["Adj Close"].iloc[-1], 2)
    print("The current price of", stock, " is: ", current_price)
    choice = input("Wanna price a call or a put ? (c/p): ")
    expiry = str(input("select the expiry date (format mm-dd-YYYY): "))
    strike_price = int(input("select the strike price: "))
    today = datetime.now()
    one_year_ago = today.replace(year=today.year - 1)
    df = web.DataReader(stock, "yahoo", one_year_ago, today)
    df = df.sort_values(by="Date")
    df = df.dropna()
    df = df.assign(close_day_before=df.Close.shift(1))
    df["returns"] = (df.Close - df.close_day_before) / df.close_day_before
    sigma = np.sqrt(252) * df["returns"].std()
    ty10y = (web.DataReader("^TNX", "yahoo")["Close"].iloc[-1]) / 100
    last_close = df["Close"].iloc[-1]
    t = (datetime.strptime(expiry, "%m-%d-%Y") - datetime.utcnow()).days / 365

    if choice == "c":
        print("The Call Price is: ", bs_call(last_close, strike_price, t, ty10y, sigma))
    if choice == "p":
        print("The Put Price is: ", bs_put(last_close, strike_price, t, ty10y, sigma))
