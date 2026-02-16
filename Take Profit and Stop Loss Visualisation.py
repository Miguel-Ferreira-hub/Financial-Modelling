# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

plt.style.use('dark_background')

# Tickers and timelines
ticker = 'AAPL'
end = dt.datetime.now()
start = end - dt.timedelta(days=18)
time = 5
plot = True
window = 5
bollinger_factor = 2

# Download data
def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, threads=False)
    df = pd.DataFrame(data)
    return df

# Define tp and sl bands for a time period (based on symmetric bollinger bands)
def bands(df, window, plot, bollinger_factor):
    # Create plot
    if plot==True:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Take Profit and Stop Loss Limits for {ticker} Close Price', fontsize=16)
        axes[0].plot(df.index, df['Close'], label='Close Price')
        axes[1].plot(df.index, df['Close'], label='Close Price')
    # Initiate band and price arrays
    bollinger_high = np.zeros(len(df['Close'].values))
    bollinger_low = np.zeros(len(df['Close'].values))
    price = np.zeros(len(df['Close'].values))
    # Slide window over prices and calculate bands
    for i in range(0,len(df['Close'].values)):
        if len(df.index[0:i]) > window and i + window < len(df.index):
            std = float(df['Close'].iloc[i-window:i].std())
            bollinger_high[i] = float(df['Close'].iloc[i]) + bollinger_factor * std
            bollinger_low[i] = float(df['Close'].iloc[i]) - bollinger_factor * std
            price[i] = float(df['Close'].iloc[i])
            # Find band locations for plot
            high_band_location = df.index[i:i+window+1]
            high_value = np.full(len(high_band_location), bollinger_high[i])
            low_band_location = df.index[i:i+window+1]
            low_value = np.full(len(low_band_location), bollinger_low[i])
            price_location = df.index[i:i+window+1]
            price_value = np.full(len(price_location), df['Close'].values[i])
            time_band_value_front = np.linspace(df['Close'].iloc[i] - bollinger_factor * std, df['Close'].iloc[i] + bollinger_factor * std, 10)
            time_band_location_front = np.full(len(time_band_value_front), df.index[i+window])
            time_band_value_back = np.linspace(df['Close'].iloc[i] - bollinger_factor * std, df['Close'].iloc[i] + bollinger_factor * std, 10)
            time_band_location_back = np.full(len(time_band_value_back), df.index[i])
            # Plot bands
            if plot == True:
                axes[0].plot(high_band_location, high_value, '--', color='green', label=f'High Bollinger Band (Take Profit)' if i == window + 1 else '')
                axes[0].plot(low_band_location, low_value, '--', color='red', label='Low Bollinger Band (Stop Loss)' if i == window + 1 else '')
                axes[0].plot(price_location, price_value, '--', color='white', label='Close Price' if i == window + 1 else '')
                axes[0].plot(time_band_location_front, time_band_value_front, '--', color='white', label='Front Time Band' if i == window + 1 else '')
                axes[0].plot(time_band_location_back, time_band_value_back, '--', color='white', label='Back Time Band' if i == window + 1 else '')
    # Show plot
    if plot == True:
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(f'{ticker} Price')
        axes[0].legend()
        plt.tight_layout()
        plt.show()
    return

# Function calls
df = get_data(ticker, start, end)

bands(df, window, plot, bollinger_factor)