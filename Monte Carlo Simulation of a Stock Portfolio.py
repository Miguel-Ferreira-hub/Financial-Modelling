import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Data Acquisition
stock_list = ['AAPL', 'MSFT', 'MSTR', 'NVDA', 'TSLA', 'AMZN']
end = dt.datetime.now()
start = end - dt.timedelta(days=300)

# Download Data
def download_data(stocks, start, end):
    data = yf.download(stocks, start=start, end=end, auto_adjust=True)
    data = data['Close']
    returns = data.pct_change().dropna()
    mean = returns.mean()
    cov = returns.cov()
    return mean, cov

# Monte Carlo Simulation Parameters
sims = 10000
time_frame = 50
initial_portfolio_balance = 1000

def simulation(sims, timeframe, mean, cov, balance):
    weights = np.random.random(len(mean))
    weights = weights/np.sum(weights)
    
    # Arrays for Data Storage
    mean = np.array(mean)
    portfolio = np.full(shape=(timeframe, sims), fill_value=0.0)

    # Monte Carlo Loops
    for i in range(0, sims):
        # Cholesky Decomposition for Daily Returns
        Z = np.random.normal(size=(timeframe, len(weights)))
        L = np.linalg.cholesky(cov)
        daily_returns = mean + (Z @ L.T)
        # Calculate Portfolio Returns
        portfolio[:,i] = np.cumprod(1 + np.dot(daily_returns, weights)) * balance
    # Expected Return (LLN)
    expected_return = portfolio.mean()
    return portfolio, expected_return

# Plot Results
mean, cov = download_data(stock_list, start=start, end=end)
portfolio, expected_return = simulation(sims=sims, timeframe=time_frame, mean=mean, cov=cov, balance=initial_portfolio_balance)

plt.style.use('dark_background')
plt.figure(figsize=(10, 5))
plt.title('Monte Carlo Simulation of a Stock Portfolio')
plt.plot(portfolio)
plt.xlabel('Days')
plt.ylabel('Portfolio Return ($)')
plt.show()

print(f'Expected Portfolio Return: {expected_return}')

# Expected Return for Investment in Individual Stocks
for stock in stock_list:
    mean, cov = download_data(stocks=stock, start=start, end=end)
    portfolio, expected_return = simulation(sims=sims, timeframe=time_frame, mean=mean, cov=cov, balance=initial_portfolio_balance)
    print(f'Expected Return for {stock}: {expected_return}')