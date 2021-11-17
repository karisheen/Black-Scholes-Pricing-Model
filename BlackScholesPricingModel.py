# Create a calculator using the black scholes model to determine the fair price for a call or put option based on six variables such as volatality, type of option, underlying stock price, time, strike, and risk-free rate.

import math
from scipy.stats import norm

class EuropeanCall:
    def call_price(
        self, asset_price, strike_price, time_to_maturity, risk_free_rate,
        volatility):

        b = math.exp(risk_free_rate * time_to_maturity)
        x1 = math.log(asset_price / strike_price)