# Black-Scholes-Merton Pricing Model in Python
## Introduction
This project is a Python implementation of the Black-Scholes-Merton model, a seminal concept in financial markets used for the theoretical pricing of options.

The script allows for two forms of data input: manual user input of financial variables, or the utilization of real-time stock data fetched using the yfinance library.

## Features
Black-Scholes-Merton Model Functions: The script includes functions for calculating the intermediary 
d1 and d2 values, as well as the final pricing for both Call and Put options.

Real-Time Stock Data: The script can fetch and utilize real-time stock data, including current price and historical data, to calculate the necessary variables for the Black-Scholes-Merton model.

Manual Input: Users have the option to manually input all variables required for the model, such as underlying asset price, strike price, time to maturity, risk-free rate, and volatility.

Interactive Experience: The script prompts users for their input and preferences, and prints the final calculated option prices.

## Libraries Used
numpy
pandas
scipy.stats
yfinance
datetime
## How to Use
Run the Python script. You will be prompted to choose whether you want to manually input the financial variables or use real-time stock data.

If you choose to use real-time data, you will be asked to specify the stock for which you want to calculate an option price. The script will fetch the current price and historical data for the stock, calculate its volatility, and use these values in the Black-Scholes-Merton model.

If you choose to input the variables manually, the script will prompt you for the necessary variables.

In both cases, you will be asked to specify whether you want to calculate the price of a Call or a Put option. The script will then print the calculated option price.

## Future Work
Future updates to this project will focus on expanding the range of financial models available, improving the accuracy of calculations, and enhancing the user interface for a better user experience.
