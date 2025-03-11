# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import date, datetime, timedelta
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Black-Scholes formula functions
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Define the app layout
app.layout = html.Div([
    html.H1("Black-Scholes Option Pricing Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        # Radio buttons to choose data input method
        dcc.RadioItems(
            id='data-choice',
            options=[
                {'label': ' Use real stock data', 'value': 'real'},
                {'label': ' Manually input variables', 'value': 'manual'}
            ],
            value='real',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            style={'margin': '10px 0'}
        ),
        
        # Div for real stock data inputs
        html.Div(
            id='real-data-input',
            children=[
                html.Div([
                    html.Label("Stock Symbol:"),
                    dcc.Input(id='stock-symbol', type='text', placeholder='e.g., AAPL', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Strike Price:"),
                    dcc.Input(id='strike-price', type='number', placeholder='e.g., 150', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Expiry Date:"),
                    dcc.DatePickerSingle(
                        id='expiry-date',
                        min_date_allowed=date.today(),
                        placeholder='Select date',
                        style={'width': '100%', 'margin-bottom': '10px'}
                    )
                ]),
                html.Div([
                    html.Label("Option Type:"),
                    dcc.RadioItems(
                        id='option-type',
                        options=[
                            {'label': ' Call', 'value': 'call'},
                            {'label': ' Put', 'value': 'put'}
                        ],
                        value='call',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    )
                ]),
            ],
            style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}
        ),
        
        # Div for manual input
        html.Div(
            id='manual-input',
            children=[
                html.Div([
                    html.Label("Underlying Price (S):"),
                    dcc.Input(id='underlying-price', type='number', placeholder='e.g., 150', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Strike Price (K):"),
                    dcc.Input(id='strike-price-manual', type='number', placeholder='e.g., 155', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Time to Expiry in Days (T):"),
                    dcc.Input(id='time-to-expiry', type='number', placeholder='e.g., 30', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Risk-Free Rate (%):"),
                    dcc.Input(id='risk-free-rate', type='number', placeholder='e.g., 2.5', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Volatility (%):"),
                    dcc.Input(id='volatility', type='number', placeholder='e.g., 25', style={'width': '100%', 'margin-bottom': '10px'})
                ]),
                html.Div([
                    html.Label("Option Type:"),
                    dcc.RadioItems(
                        id='option-type-manual',
                        options=[
                            {'label': ' Call', 'value': 'call'},
                            {'label': ' Put', 'value': 'put'}
                        ],
                        value='call',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    )
                ]),
            ],
            style={'display': 'none', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}
        ),
        
        html.Br(),
        html.Button('Calculate', id='calculate-button', 
                   style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 
                          'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    
    # Results area with visualizations
    html.Div(id='output-area', style={'width': '55%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1200px'})

# Callback to toggle between input forms
@app.callback(
    [Output('real-data-input', 'style'),
     Output('manual-input', 'style')],
    [Input('data-choice', 'value')]
)
def toggle_input_forms(choice):
    if choice == 'real':
        return {'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}

# Callback for calculating option price with visualizations
@app.callback(
    Output('output-area', 'children'),
    Input('calculate-button', 'n_clicks'),
    [State('data-choice', 'value'),
     State('stock-symbol', 'value'),
     State('strike-price', 'value'),
     State('expiry-date', 'date'),
     State('option-type', 'value'),
     State('underlying-price', 'value'),
     State('strike-price-manual', 'value'),
     State('time-to-expiry', 'value'),
     State('risk-free-rate', 'value'),
     State('volatility', 'value'),
     State('option-type-manual', 'value')]
)
def calculate_option_price(n_clicks, choice, stock_symbol, strike_price_real, expiry_date, option_type_real,
                           underlying_price, strike_price_manual, time_to_expiry, risk_free_rate, volatility,
                           option_type_manual):
    if n_clicks is None:
        return ""
    
    if choice == 'real':
        # Validate inputs
        if not stock_symbol or not strike_price_real or not expiry_date:
            return html.Div("Please fill in all required fields.", style={'color': 'red'})
        
        # Fetch and process real stock data
        try:
            # Get stock data
            stock_data = yf.Ticker(stock_symbol)
            current_price = stock_data.history(period="1d")['Close'].iloc[-0]
            
            # Calculate time to expiry
            today = date.today()
            expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            days_to_expiry = (expiry - today).days
            if days_to_expiry <= 0:
                return html.Div("Expiry date must be in the future.", style={'color': 'red'})
            
            T = days_to_expiry / 365.0  # Convert days to years
            
            # Get historical volatility (using 1 year of data)
            hist_data = stock_data.history(period="1y")
            returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            sigma = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Use a default risk-free rate (e.g., 10-year Treasury yield)
            r = 0.025  # 2.5% as an example
            
            # Calculate option price
            strike_price_real = float(strike_price_real)
            if option_type_real == 'call':
                option_price = bs_call(current_price, strike_price_real, T, r, sigma)
            else:
                option_price = bs_put(current_price, strike_price_real, T, r, sigma)
            
            # Generate price vs strike price graph data
            strikes = np.linspace(current_price * 0.7, current_price * 1.3, 50)
            call_prices = [bs_call(current_price, k, T, r, sigma) for k in strikes]
            put_prices = [bs_put(current_price, k, T, r, sigma) for k in strikes]
            
            # Generate price vs time to expiry graph data
            times = np.linspace(1/365, 2, 50)  # 1 day to 2 years
            time_call_prices = [bs_call(current_price, strike_price_real, t, r, sigma) for t in times]
            time_put_prices = [bs_put(current_price, strike_price_real, t, r, sigma) for t in times]
            
            # Format and return the result with visualizations
            return html.Div([
                html.Div([
                    html.H3(f"{option_type_real.capitalize()} Option Price: ${option_price:.2f}", 
                           style={'color': '#2c3e50', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.P(f"Underlying Price (S): ${current_price:.2f}"),
                            html.P(f"Strike Price (K): ${strike_price_real:.2f}"),
                            html.P(f"Time to Expiry (T): {days_to_expiry} days"),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        html.Div([
                            html.P(f"Risk-Free Rate (r): {r*100:.2f}%"),
                            html.P(f"Volatility (σ): {sigma*100:.2f}%"),
                            html.P(f"Option Type: {option_type_real.capitalize()}"),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'})
                ]),
                
                html.Div([
                    html.H4("Option Price vs Strike Price", style={'textAlign': 'center'}),
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Scatter(x=strikes, y=call_prices, mode='lines', name='Call Price'),
                                go.Scatter(x=strikes, y=put_prices, mode='lines', name='Put Price'),
                                go.Scatter(x=[strike_price_real], y=[option_price], mode='markers', 
                                          marker=dict(size=10, color='red'), name='Current Option')
                            ],
                            'layout': go.Layout(
                                xaxis={'title': 'Strike Price'},
                                yaxis={'title': 'Option Price'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                hovermode='closest'
                            )
                        }
                    )
                ]),
                
                html.Div([
                    html.H4("Option Price vs Time to Expiry", style={'textAlign': 'center'}),
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Scatter(x=[t*365 for t in times], y=time_call_prices, mode='lines', name='Call Price'),
                                go.Scatter(x=[t*365 for t in times], y=time_put_prices, mode='lines', name='Put Price'),
                                go.Scatter(x=[days_to_expiry], y=[option_price], mode='markers', 
                                          marker=dict(size=10, color='red'), name='Current Option')
                            ],
                            'layout': go.Layout(
                                xaxis={'title': 'Days to Expiry'},
                                yaxis={'title': 'Option Price'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                hovermode='closest'
                            )
                        }
                    )
                ])
            ])
        except Exception as e:
            return html.Div(f"An error occurred: {e}", style={'color': 'red'})
    
    elif choice == 'manual':
        # Validate inputs
        if not underlying_price or not strike_price_manual or not time_to_expiry or not risk_free_rate or not volatility:
            return html.Div("Please fill in all required fields.", style={'color': 'red'})
        
        # Process manual inputs
        try:
            S = float(underlying_price)
            K = float(strike_price_manual)
            T_days = float(time_to_expiry)
            T = T_days / 365.0  # Convert days to years
            r = float(risk_free_rate) / 100.0  # Convert percentage to decimal
            sigma = float(volatility) / 100.0  # Convert percentage to decimal
            
            # Calculate option price
            if option_type_manual == 'call':
                option_price = bs_call(S, K, T, r, sigma)
            else:
                option_price = bs_put(S, K, T, r, sigma)
            
            # Generate price vs strike price graph data
            strikes = np.linspace(S * 0.7, S * 1.3, 50)
            call_prices = [bs_call(S, k, T, r, sigma) for k in strikes]
            put_prices = [bs_put(S, k, T, r, sigma) for k in strikes]
            
            # Generate price vs time to expiry graph data
            times = np.linspace(1/365, 2, 50)  # 1 day to 2 years
            time_call_prices = [bs_call(S, K, t, r, sigma) for t in times]
            time_put_prices = [bs_put(S, K, t, r, sigma) for t in times]
            
            # Format and return the result with visualizations
            return html.Div([
                html.Div([
                    html.H3(f"{option_type_manual.capitalize()} Option Price: ${option_price:.2f}", 
                           style={'color': '#2c3e50', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.P(f"Underlying Price (S): ${S:.2f}"),
                            html.P(f"Strike Price (K): ${K:.2f}"),
                            html.P(f"Time to Expiry (T): {T_days} days"),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        html.Div([
                            html.P(f"Risk-Free Rate (r): {r*100:.2f}%"),
                            html.P(f"Volatility (σ): {sigma*100:.2f}%"),
                            html.P(f"Option Type: {option_type_manual.capitalize()}"),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'})
                ]),
                
                html.Div([
                    html.H4("Option Price vs Strike Price", style={'textAlign': 'center'}),
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Scatter(x=strikes, y=call_prices, mode='lines', name='Call Price'),
                                go.Scatter(x=strikes, y=put_prices, mode='lines', name='Put Price'),
                                go.Scatter(x=[K], y=[option_price], mode='markers', 
                                          marker=dict(size=10, color='red'), name='Current Option')
                            ],
                            'layout': go.Layout(
                                xaxis={'title': 'Strike Price'},
                                yaxis={'title': 'Option Price'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                hovermode='closest'
                            )
                        }
                    )
                ]),
                
                html.Div([
                    html.H4("Option Price vs Time to Expiry", style={'textAlign': 'center'}),
                    dcc.Graph(
                        figure={
                            'data': [
                                go.Scatter(x=[t*365 for t in times], y=time_call_prices, mode='lines', name='Call Price'),
                                go.Scatter(x=[t*365 for t in times], y=time_put_prices, mode='lines', name='Put Price'),
                                go.Scatter(x=[T_days], y=[option_price], mode='markers', 
                                          marker=dict(size=10, color='red'), name='Current Option')
                            ],
                            'layout': go.Layout(
                                xaxis={'title': 'Days to Expiry'},
                                yaxis={'title': 'Option Price'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                hovermode='closest'
                            )
                        }
                    )
                ])
            ])
        except Exception as e:
            return html.Div(f"An error occurred: {e}", style={'color': 'red'})
    else:
        return "Invalid data choice selected."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)