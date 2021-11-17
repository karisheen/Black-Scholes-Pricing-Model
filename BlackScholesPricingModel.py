# Create a calculator using the black scholes model to determine the fair price for a call or put option 
# based on six variables such as volatality, type of option, underlying stock price, time, strike, and risk-free rate.

import math
from scipy.stats import norm

class EuropeanCall:
    def call_price(
        self, asset_price, strike_price, time_to_maturity, risk_free_rate,
        volatility):

        b = math.exp(risk_free_rate * time_to_maturity)
        x1 = math.log(asset_price / (b*strike_price)) + .5 *(volatility * volatility * volatility) * time_to_maturity
        x1 = x1 / (volatility * (time_to_maturity ** .5))
        z1 = norm.cdf(x1)
        z1 = z1 * asset_price
        x2 = math.log(asset_price / (b*strike_price)) - .5 *(volatility * volatility * volatility) * time_to_maturity
        x2 = x2 / (volatility * (time_to_maturity ** .5))
        z2 = norm.cdf(x2)
        z2 = b * strike_price * z2
        return z1 - z2

        def __init__(self, asset_price, strike_price, time_to_maturity, risk_free_rate, volatility):
            self.asset_price = asset_price
            self.strike_price = strike_price
            self.time_to_maturity = time_to_maturity
            self.risk_free_rate = risk_free_rate
            self.volatility = volatility
            self.price = self.call_price(asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)

class EuropeanPut:
    def put_price(
        self, asset_price, strike_price, time_to_maturity, risk_free_rate,
        volatility):

        