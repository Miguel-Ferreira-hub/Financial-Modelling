import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
import numpy as np
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
import pandas as pd
import os

plt.style.use('dark_background')

# Initial Sampling Parameters
filename = r'NVDA_MeanReversion.npz'
past_days = 10
end = dt.datetime.now()
start = end - dt.timedelta(days=past_days)
interval = '15m'
ticker = 'NVDA'

# Load Sample Data - Example Mean Reverting Regime
def load_data(filename):
    directory = r'C:\Users\Miguel\Desktop\Interactive Brokers'
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        file = np.load(path)
        data = file['X_data']
        print(f'Data Loaded From {path}')
        return data
    else:
        print('No Filepath Found')
    return None

# Kalman Filter Function
def Kalman(x,P,R,close_price,mu,theta,sigma):
    dt = 15/(60*24)
    filtered_values = []
    for price in close_price:
        # Prediction
        z = price
        # OU state transition
        a = np.exp(-theta * dt)
        # Process noise variance implied by OU
        Q = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*dt))
        x_pred = mu + a * (x - mu)
        P_pred = a*a * P + Q
        # Update
        K = P_pred / (P_pred + R)
        x_new = x_pred + (z-x_pred)*K
        P_new = (1-K)*P_pred
        filtered_values.append(x_new)
        P = P_new
        x = x_new

    return filtered_values

# Fit OU Process to Data
def load_ticker(ticker, interval):
    if isinstance(ticker, str):
        data = yf.download(tickers=ticker,start=start,end=end,interval=interval)
        close = data['Close'].squeeze()
    else:
        close = pd.DataFrame(ticker).squeeze()
    
    # Calibration of OU Process
    df = pd.DataFrame({'delta': close.diff(),'lag_close': close.shift(1)}).dropna()
    X = sm.add_constant(df['lag_close'])
    y = df['delta']

    model = sm.OLS(y, X).fit()

    alpha = model.params['const']
    beta = model.params['lag_close']
    var_eps = np.var(model.resid)
    close_price = close.values

    dt = 15/(60*24)
    phi = 1 + beta
    theta = -np.log(phi)/dt
    mu = alpha/(1-phi)
    sigma = np.sqrt((2*theta*var_eps)/(1-np.exp(-2*theta*dt)))
        
    # Print Statements
    print(model.summary())
    print(f'alpha: {alpha:.2f}')
    print(f'beta: {beta:.2f}')
    print(f'error variance: {var_eps:.2f}')
    print(f'theta: {theta:.2f}')
    print(f'mu: {mu:.2f}')
    print(f'sigma: {sigma:.2f}')

    return close_price, mu, theta, sigma, var_eps

example_data = load_data(filename)

close_price, mu, theta, sigma, var_eps = load_ticker(ticker, interval)

# Model Parameters
S0 = close_price[0]
low = 0.1
high = 1000000
initR = 500
x = S0
P = 100

kalman_mean = np.mean(Kalman(x,P,initR,close_price,mu,theta,sigma))

# Display
fig, ax = plt.subplots(figsize=(8,6))
fig.subplots_adjust(left=0.25, bottom=0.25)

line, = ax.plot(Kalman(x,P,initR,close_price,mu,theta,sigma), lw=2, label='Kalman')
l, = ax.plot(close_price,label=f'{ticker} Close Price')
ax.set_title(f'Kalman Mean Reversion on {ticker} Close Price')
mean_linek, = ax.plot([0, len(close_price)],[kalman_mean, kalman_mean],color='red',label='Kalman Mean')
ax.set_xlabel('Time')
ax.set_ylabel(f'{ticker} Close Price ($)')
mean = [mu,mu]
muline, = ax.plot([0, len(close_price)],[mu, mu],label='OU Mean',color='purple')
meanline, = ax.plot([0, len(close_price)],[np.mean(close_price),np.mean(close_price)],label='Actual Mean',color='orange')
ax.legend()

axs = fig.add_axes((0.25, 0.1, 0.65, 0.03))
slider = Slider(ax=axs,label='Measurement',valmin=low,valmax=high,valinit=initR)

slider.valtext.set_text('Process')

slider.ax.set_xscale("log")

# Slider Update
def update(val):
    x0 = close_price[0]
    filtered = Kalman(x0, P, slider.val,close_price, mu, theta, sigma)
    line.set_ydata(filtered)
    kalman_mean = np.mean(filtered)
    mean_linek.set_ydata([kalman_mean, kalman_mean])
    slider.valtext.set_text('Process')
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

# Submit Ticker
def submit(new_ticker):
    global ticker, close_price, mu, theta, sigma
    new_ticker = new_ticker.upper()
    ticker = new_ticker
    if new_ticker == 'SAMPLE':
        close_price, mu, theta, sigma, _ = load_ticker(example_data, interval)
        new_ticker = 'Sample (NVDA Mean Reversion Regime)'
    else:
        close_price, mu, theta, sigma, _ = load_ticker(new_ticker, interval)
    x0 = close_price[0]
    filtered = Kalman(x0,P,slider.val,close_price,mu,theta,sigma)
    line.set_data(np.arange(len(filtered)), filtered)
    l.set_data(np.arange(len(close_price)),close_price)
    l.set_label(f'{new_ticker} Close Price')
    muline.set_data([0, len(close_price)],[mu,mu])
    meanline.set_data([0, len(close_price)],[np.mean(close_price),np.mean(close_price)])
    mean_linek.set_data([0, len(close_price)],[np.mean(filtered), np.mean(filtered)])
    ax.set_title(f'Kalman Mean Reversion on {new_ticker} Close Price')
    ax.set_ylabel(f'{new_ticker} Close Price ($)')
    ax.relim()
    ax.autoscale_view()
    ax.legend()
    fig.canvas.draw_idle()
    text_box.set_val('')

# Buttons
resetax = fig.add_axes((0.8, 0.025, 0.1, 0.04))
exitax = fig.add_axes((0.25, 0.025, 0.1, 0.04))
tickerax = fig.add_axes((0.10, 0.025, 0.05, 0.04))
tickerax.set_title('Ticker')
button = Button(resetax, 'Reset', color='black', hovercolor='0.400')
buttonExit = Button(exitax, 'Exit', color='black', hovercolor='0.400')
text_box = TextBox(tickerax, '', textalignment='center', color='black', hovercolor='0.400')
resetax.spines[:].set_color("0.5")
exitax.spines[:].set_color("0.5")
tickerax.spines[:].set_color("0.5")
text_box.on_submit(submit)
text_box.set_val('')

# Reset Chart
def reset(event):
    global start, end, interval, ticker
    end = dt.datetime.now()
    start = end - dt.timedelta(days=10)
    ticker = 'NVDA'
    interval = '15m'
    slider.reset()
    text_box.set_val('')

button.on_clicked(reset)

# Exit Display
def exit(event):
    plt.close(fig)
buttonExit.on_clicked(exit)

slider.on_changed(update)

figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()

plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.show()
