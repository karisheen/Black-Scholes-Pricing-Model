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
    html.H1("Black-Scholes Option Pricing Dashboard"),
    
    # Radio buttons to choose data input method
    dcc.RadioItems(
        id='data-choice',
        options=[
            {'label': ' Use real stock data', 'value': 'real'},
            {'label': ' Manually input variables', 'value': 'manual'}
        ],
        value='real',
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
    
    # Div for real stock data inputs
    html.Div(
        id='real-data-input',
        children=[
            html.Br(),
            dcc.Input(id='stock-symbol', type='text', placeholder='Stock Symbol'),
            html.Br(),
            dcc.Input(id='strike-price', type='number', placeholder='Strike Price'),
            html.Br(),
            dcc.Input(id='expiry-date', type='text', placeholder='Expiry Date (mm-dd-yyyy)'),
            html.Br(),
            dcc.RadioItems(
                id='option-type',
                options=[
                    {'label': ' Call', 'value': 'call'},
                    {'label': ' Put', 'value': 'put'}
                ],
                value='call',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
        ]
    ),
    
    # Div for manual inputs
    html.Div(
        id='manual-input',
        children=[
            html.Br(),
            dcc.Input(id='underlying-price', type='number', placeholder='Underlying Price (S)'),
            html.Br(),
            dcc.Input(id='strike-price-manual', type='number', placeholder='Strike Price (K)'),
            html.Br(),
            dcc.Input(id='time-to-expiry', type='number', placeholder='Time to Expiry in Days (T)'),
            html.Br(),
            dcc.Input(id='risk-free-rate', type='number', placeholder='Risk-Free Rate (%)'),
            html.Br(),
            dcc.Input(id='volatility', type='number', placeholder='Volatility (%)'),
            html.Br(),
            dcc.RadioItems(
                id='option-type-manual',
                options=[
                    {'label': ' Call', 'value': 'call'},
                    {'label': ' Put', 'value': 'put'}
                ],
                value='call',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
        ],
        style={'display': 'none'}  # Initially hidden
    ),
    
    html.Br(),
    html.Button('Calculate', id='calculate-button'),
    html.Div(id='output-area')
])

# Callback to hide/show input forms based on selection
@app.callback(
    Output('real-data-input', 'style'),
    Output('manual-input', 'style'),
    Input('data-choice', 'value')
)
def toggle_input_forms(choice):
    if choice == 'real':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback for calculating option price
@app.callback(
    Output('output-area', 'children'),
    Input('calculate-button', 'n_clicks'),
    State('data-choice', 'value'),
    # Real data inputs
    State('stock-symbol', 'value'),
    State('strike-price', 'value'),
    State('expiry-date', 'value'),
    State('option-type', 'value'),
    # Manual inputs
    State('underlying-price', 'value'),
    State('strike-price-manual', 'value'),
    State('time-to-expiry', 'value'),
    State('risk-free-rate', 'value'),
    State('volatility', 'value'),
    State('option-type-manual', 'value')
)
def calculate_option_price(n_clicks, choice, stock_symbol, strike_price_real, expiry_date, option_type_real,
                           underlying_price, strike_price_manual, time_to_expiry, risk_free_rate, volatility,
                           option_type_manual):
    if n_clicks is None:
        return ""
    
    if choice == 'real':
        # Validate inputs for real data
        if not all([stock_symbol, strike_price_real, expiry_date, option_type_real]):
            return "Please provide all required inputs for real stock data."
        
        # Fetch and process real stock data
        try:
            # Fetch current price
            stock_data = yf.Ticker(stock_symbol)
            hist = stock_data.history(period="1d")
            if hist.empty:
                return f"No data found for symbol {stock_symbol}."
            current_price = hist['Close'].iloc[-1]
            
            # Calculate time to expiry
            expiry_datetime = datetime.strptime(expiry_date, "%m-%d-%Y")
            days_to_expiry = (expiry_datetime - datetime.now()).days
            if days_to_expiry <= 0:
                return "The expiry date must be in the future."
            T = days_to_expiry / 365.0
            
            # Risk-free rate (you can fetch this data or use a default value)
            r = 0.02  # Assume 2% risk-free rate
            
            # Calculate volatility
            historical_data = stock_data.history(period="1y")
            if historical_data.empty:
                return f"Not enough historical data to calculate volatility for {stock_symbol}."
            historical_data['returns'] = historical_data['Close'].pct_change()
            sigma = np.sqrt(252) * historical_data['returns'].std()
            
            # Validate sigma
            if sigma <= 0:
                return "Calculated volatility is non-positive. Please check the stock data."
            
            # Calculate option price
            strike_price_real = float(strike_price_real)
            if option_type_real == 'call':
                option_price = bs_call(current_price, strike_price_real, T, r, sigma)
            else:
                option_price = bs_put(current_price, strike_price_real, T, r, sigma)
            
            # Format and return the result
            return html.Div([
                html.H3(f"{option_type_real.capitalize()} Option Price: ${option_price:.2f}"),
                html.P(f"Underlying Price (S): ${current_price:.2f}"),
                html.P(f"Strike Price (K): ${strike_price_real:.2f}"),
                html.P(f"Time to Expiry (T): {days_to_expiry} days"),
                html.P(f"Risk-Free Rate (r): {r*100:.2f}%"),
                html.P(f"Volatility (σ): {sigma*100:.2f}%")
            ])
        except Exception as e:
            return f"An error occurred: {e}"
    
    elif choice == 'manual':
        # Validate inputs for manual data
        if not all([underlying_price, strike_price_manual, time_to_expiry, risk_free_rate, volatility, option_type_manual]):
            return "Please provide all required inputs for manual data."
        
        # Process manual inputs
        try:
            S = float(underlying_price)
            K = float(strike_price_manual)
            T_days = float(time_to_expiry)
            T = T_days / 365.0  # Convert days to years
            r = float(risk_free_rate) / 100  # Convert percentage to decimal
            sigma = float(volatility) / 100  # Convert percentage to decimal
            
            # Validate inputs
            if T <= 0:
                return "Time to expiry must be greater than zero."
            if sigma <= 0:
                return "Volatility must be positive."
            if S <= 0 or K <= 0:
                return "Prices must be positive."
            
            # Calculate option price
            if option_type_manual == 'call':
                option_price = bs_call(S, K, T, r, sigma)
            else:
                option_price = bs_put(S, K, T, r, sigma)
            
            # Format and return the result
            return html.Div([
                html.H3(f"{option_type_manual.capitalize()} Option Price: ${option_price:.2f}"),
                html.P(f"Underlying Price (S): ${S:.2f}"),
                html.P(f"Strike Price (K): ${K:.2f}"),
                html.P(f"Time to Expiry (T): {T_days} days"),
                html.P(f"Risk-Free Rate (r): {r*100:.2f}%"),
                html.P(f"Volatility (σ): {sigma*100:.2f}%")
            ])
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        return "Invalid data choice selected."

if __name__ == '__main__':
    app.run_server(debug=True)


