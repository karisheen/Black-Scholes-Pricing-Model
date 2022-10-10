# Create a calculator using the black scholes model to determine the fair price for a call or put option 
# based on six variables such as volatality, type of option, underlying stock price, time, strike, and risk-free rate.

import math
from scipy.stats import norm

# Define the variables
# S = underlying asset price
# K = strike price
# T = time to maturity
# r = risk-free rate
# sigma = volatility

class EuropeanCall:

    def call_price(
        self, S, sigma, K,
        T, r
            ):
        b = math.exp(-r*T)
        x1 = math.log(S/(b*K)) + .5*(sigma*sigma)*T
        x1 = x1/(sigma*(T**.5))
        z1 = norm.cdf(x1)
        z1 = z1*S
        x2 = math.log(S/(b*K)) - .5*(sigma*sigma)*T
        x2 = x2/(sigma*(T**.5))
        z2 = norm.cdf(x2)
        z2 = b*K*z2
        return z1 - z2

    def __init__(
        self, S, sigma, K,
        T, r
            ):
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r
        self.price = self.call_price(S, sigma, K, T, r)
        

class EuropeanPut:

    def put_price(
        self, S, sigma, K,
        T, r
            ):
        b = math.exp(-r*T)
        x1 = math.log((b*K)/S) + .5*(sigma*sigma)*T
        x1 = x1/(sigma*(T**.5))
        z1 = norm.cdf(x1)
        z1 = b*K*z1
        x2 = math.log((b*K)/S) - .5*(sigma*sigma)*T
        x2 = x2/(sigma*(T**.5))
        z2 = norm.cdf(x2)
        z2 = S*z2
        return z1 - z2

    def __init__(
        self, S, sigma, K,
        T, r
            ):
        self.S = S
        self.sigma = sigma
        self.K = K
        self.T = T
        self.r = r
        self.price = self.put_price(S, sigma, K, T, r)

ec = EuropeanCall(100, .3, 90, 30, .01)
print ("The price of the call option is: ", ec.price)

ep = EuropeanPut(100, .3, 90, 30, .01)
print ("The price of the put option is: ", ep.price)
