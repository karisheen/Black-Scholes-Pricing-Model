# Import necessary libraries
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
from datetime import datetime
import yfinance as yf
import numpy as np

# Import the Black Scholes functions from your model file
from BlackScholesPricingModel import bs_call, bs_put

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Black Scholes Option Pricing Dashboard"),
    dcc.RadioItems(
        id='data-choice',
        options=[
            {'label': 'Use real stock data', 'value': 'real'},
            {'label': 'Manually input variables', 'value': 'manual'}
        ],
        value='real'
    ),
    html.Div(id='input-area'),
    html.Button('Calculate', id='calculate-button'),
    html.Div(id='output-area')
])

# Callback for dynamic input fields
@app.callback(
    Output('input-area', 'children'),
    Input('data-choice', 'value')
)
def update_input_area(choice):
    if choice == 'real':
        return html.Div([
            dcc.Input(id='stock-symbol', type='text', placeholder='Stock Symbol'),
            dcc.Input(id='strike-price', type='number', placeholder='Strike Price'),
            dcc.Input(id='expiry-date', type='text', placeholder='Expiry Date (mm-dd-yyyy)'),
            dcc.RadioItems(
                id='option-type',
                options=[
                    {'label': 'Call', 'value': 'call'},
                    {'label': 'Put', 'value': 'put'}
                ],
                value='call'
            )
        ])
    else:
        return html.Div([
            dcc.Input(id='underlying-price', type='number', placeholder='Underlying Price (S)'),
            dcc.Input(id='strike-price-manual', type='number', placeholder='Strike Price (K)'),
            dcc.Input(id='time-to-expiry', type='number', placeholder='Time to Expiry in Days (T)'),
            dcc.Input(id='risk-free-rate', type='number', placeholder='Risk-Free Rate (%)'),
            dcc.Input(id='volatility', type='number', placeholder='Volatility (%)'),
            dcc.RadioItems(
                id='option-type-manual',
                options=[
                    {'label': 'Call', 'value': 'call'},
                    {'label': 'Put', 'value': 'put'}
                ],
                value='call'
            )
        ])

# Callback for calculating option price
@app.callback(
    Output('output-area', 'children'),
    Input('calculate-button', 'n_clicks'),
    State('data-choice', 'value'),
    State('stock-symbol', 'value'),
    State('strike-price', 'value'),
    State('expiry-date', 'value'),
    State('option-type', 'value'),
    State('underlying-price', 'value'),
    State('strike-price-manual', 'value'),
    State('time-to-expiry', 'value'),
    State('risk-free-rate', 'value'),
    State('volatility', 'value'),
    State('option-type-manual', 'value')
)
def calculate_option_price(n_clicks, choice, symbol, strike, expiry, opt_type,
                           S, K_manual, T_days, r, vol, opt_type_manual):
    if n_clicks is None:
        return ''
    
    try:
        if choice == 'real':
            data = yf.download(symbol, period="1y")
            S = data["Adj Close"].iloc[-1]
            T = (datetime.strptime(expiry, "%m-%d-%Y") - datetime.now()).days / 365
            sigma = np.sqrt(252) * data["returns"].std()
            r = 0.02  # Fixed risk-free rate

            if opt_type == 'call':
                price = bs_call(S, strike, T, r, sigma)
            else:
                price = bs_put(S, strike, T, r, sigma)

            return f'The calculated {opt_type} option price for {symbol} is: {price:.2f}'

        else:
            T = T_days / 365  # Convert days to years
            r /= 100  # Convert percentage to decimal
            vol /= 100  # Convert percentage to decimal

            if opt_type_manual == 'call':
                price = bs_call(S, K_manual, T, r, vol)
            else:
                price = bs_put(S, K_manual, T, r, vol)

            return f'The calculated {opt_type_manual} option price is: {price:.2f}'

    except Exception as e:
        return f'Error: {e}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
