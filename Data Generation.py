import numpy as np # Imports
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

# This script is used to produce data for backtests based on the statistical properties of past samples

plt.style.use('dark_background')

tickers = ['MSTR'] # Example tickers
end = dt.datetime.now() # Extract samples to current date
start = end - dt.timedelta(days=365) # Begin 
n_steps = 100 # Number of new datapoints simulated

def get_data(ticker, start, end):
    data = yf.download(ticker, start, end) # Download data
    returns = np.log(data['Close']).diff().dropna() # Log returns
    mu = returns.mean() * 252 # Annualised mean
    sigma = returns.std() * np.sqrt(252) # Annualised standard deviation (volatility)
    S0 = data['Close'].iloc[-1] # Beginning data generation from the last price
    return mu, sigma, S0

def simulate_path(mu, sigma, S0, n_steps):
    mu, sigma, S0 = float(mu), float(sigma), float(S0)
    T = 1
    dt = T / n_steps # Change in step
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))  # Wiener process (standard brownian motion)
    t = np.linspace(dt, T, n_steps) # Timestep domain
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W) # Geometric brownian motion
    return t, S

def simulated_data(tickers, start, end, n_steps):
    # Genearte data and plot
    fig, axes = plt.subplots(len(tickers), 1, figsize=(12, 6))
    fig.suptitle(f'Simulated Data', fontsize=16)
    if len(tickers) == 1:
        axes = [axes]
    for i in range(0, len(tickers)):
        mu, sigma, S0 = get_data(tickers[i], start, end)
        t, S = simulate_path(mu, sigma, S0, n_steps)
        axes[i].plot(t, S)
        axes[i].set_ylabel(f'{tickers[i]} Close Price ($)')
    axes[len(tickers)-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
    return

simulated_data(tickers, start, end, n_steps)