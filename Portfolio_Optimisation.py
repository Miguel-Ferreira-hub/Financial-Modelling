import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.default import DefaultMultiObjectiveTermination
import scipy.optimize as sp
from tqdm import tqdm

plt.style.use('dark_background')

# Sampling Period and Parameters
end = dt.datetime.now()
start = end - dt.timedelta(days=100)
interval = '1d'
tickers = ['NVTS', 'NVDA', 'AMD', 'AAPL' ,'MSFT', 'AMZN', 'GOOG', 'PLTR', 'MSTR', 'MU', 'RTX', 'LMT', 'BA', 'GD', 'DAL', 'NOC', 'LDOS', 'GE', 'HWM', 'TDG', 'JPM', 'HSBC', 'LYG', 'BRK-B', 'BAC', 'WFC', 'C', 'MS', 'GS', 'MA']
tickers = tickers[:len(tickers)//2]

def ret(tickers,start,end,interval):
    df = pd.DataFrame()
    expected_returns = []
    for ticker in tickers:
        data = yf.download(tickers=ticker,start=start,end=end,interval=interval)
        close = data['Close']
        df[f'{ticker}'] = close
        returns = close.pct_change().dropna()
        expected_return = returns.mean().item() * 252
        expected_returns.append(expected_return)
    cov = df.pct_change().dropna().cov() * 252
    return df, expected_returns, cov

def portfolio(expected_returns, cov):
    rf = 0.04
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    portfolio_returns = weights @ expected_returns
    risk = np.sqrt(weights @ cov @ weights)
    sharpe = (portfolio_returns - rf) / risk
    return portfolio_returns, risk, sharpe, weights

# Efficient Frontier Using Monte Carlo Simulation
n_sims = 5000000

df, expected_returns, cov = ret(tickers=tickers,start=start,end=end,interval=interval)

returns_list = []
risk_list = []
sharpe_list = []
weights_list = []

for i in tqdm(range(n_sims),desc='Efficient Frontier via Monte Carlo Simulation'):
    r, s, sh, weights = portfolio(expected_returns, cov)
    returns_list.append(r)
    risk_list.append(s)
    sharpe_list.append(sh)
    weights_list.append(weights.tolist())

# Monte Carlo Results Plot
plt.figure(figsize=(8,6))
plt.scatter(risk_list,returns_list,c=sharpe_list,cmap="viridis")
plt.title('Efficient Frontier via Monte Carlo Simulation')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Annualised Expected Return")
plt.colorbar(label="Sharpe Ratio")
plt.grid(alpha=0.3)

idx = max(np.argsort(sharpe_list))
weights_max_sharpe = weights_list[idx]

# Efficient Frontier Using Genetic Algorithm (NSGA-II) Global Optimisation for Pareto Front
def objective(weights,expected_returns,cov): # Returns negative portfolio returns so that both objectives can be minimised by GA
    portfolio_returns = weights @ expected_returns
    risk = np.sqrt(weights @ cov @ weights)
    return -portfolio_returns, risk

class ProblemWrapper(Problem):

    def _evaluate(self,designs,out,*args,**kwargs):
        res = []
        H = []

        for design in designs:
            res.append(objective(design,expected_returns=expected_returns,cov=cov))
            H.append(np.sum(design) - 1)

        out["F"] = np.array(res)
        out["H"] = np.array(H).reshape(-1, 1)

problem = ProblemWrapper(n_var=len(tickers),n_obj=2,xl=[0]*len(tickers),xu=[1]*len(tickers),n_ieq_constr=0,n_eq_constr=1)

algorithm = NSGA2(pop_size=10000,eliminate_duplicates=False)

termination = DefaultMultiObjectiveTermination(xtol=1e-10,cvtol=1e-10,ftol=1e-10,period=1000,n_max_gen=1000,n_max_evals=10000000)

results = minimize(problem=problem,algorithm=algorithm,termination=termination,seed=1,verbose=True)

res_data = results.F.T

# Compute Sharpe
rf = 0.04
sharpe_opt = []

for i in range(len(res_data[0])):
    sharpe_opt.append((-res_data[0][i] - rf) / res_data[1][i])

# NSGA2 Portfolio Optimsation Plot
plt.figure(figsize=(8,6))
plt.scatter(res_data[1],-res_data[0],c=sharpe_opt,cmap='viridis')
plt.title('Efficient Frontier via NSGA2 Optimisation')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Annualised Expected Return")
plt.colorbar(label="Sharpe Ratio")
plt.grid(alpha=0.3)

# Optimisation for Volatility Drag: geometric mean -> arithmetic mean - 1/2vol^2
def vol(weights,df,cov):
    retdf = df.pct_change().dropna()
    l = retdf.mean().tolist()
    mu = (weights @ np.array(l))
    sigma = np.sqrt(weights @ cov @ weights)
    volatility_drag = 0.5*sigma*sigma
    obj = mu - volatility_drag
    return -obj

def constraint(weights):
    con = weights.sum() - 1
    return con

x0 = [0.1] * len(tickers)
b = []
for i in range(len(tickers)):
    b.append((0,1))
bounds = tuple(b)

con = {'type':'eq','fun':constraint}
cons = [con]

sol = sp.minimize(
    vol,
    x0,
    args=(df, cov),
    method='SLSQP',
    bounds=bounds,
    constraints=cons
)

pr, vol = objective(sol.x,expected_returns=expected_returns,cov=cov)

# Comparison Between Methods
plt.figure(figsize=(8,6))
sc = plt.scatter(risk_list,returns_list,c=sharpe_list,cmap="viridis")
plt.scatter(res_data[1],-res_data[0],color='red',label='NSGA2 Pareto Front')
plt.scatter(vol,-pr,color='gold',label=r'Minimise Volatility Drag: $\mu - \frac{1}{2}\sigma^2$')
plt.title('Efficient Frontier')
plt.xlabel("Portfolio Volatility")
plt.ylabel("Annualised Expected Return")
plt.colorbar(sc,label="Sharpe Ratio")
plt.legend()
plt.grid(alpha=0.3)

# Construct Portfolios
def constructPortfolio(weights, df, tickers, investment):
    portfolio = None
    for ticker, weight in zip(tickers, weights):
        returns = df[ticker].pct_change().dropna()
        growth = (1 + returns).cumprod()
        value = investment * weight * growth
        if portfolio is None:
            portfolio = value
        else:
            portfolio = portfolio.add(value, fill_value=0)
    return portfolio

investment = 1000000

# weights_max_sharpe = [0.11801670057730448, 0.08521781081915017, 0.13285334841543528, 0.029742251161869473, 0.11112349790650732, 0.030366568576320176, 0.084422990880403, 0.03894517963221414, 0.022841788517246102, 0.10410082668476481, 0.019751180490001234, 0.10585469716045175, 0.01479961207430556, 0.0888738399891918, 0.013089707114834726]

# weights2 = [4.12911200e-18, 9.74315158e-02, 0.00000000e+00, 2.15932378e-01, 1.30704863e-01, 4.25648581e-02, 6.54120850e-02, 1.06891216e-17, 1.01534845e-17, 4.83063793e-02, 1.03073001e-01, 1.02123206e-01, 0.00000000e+00, 1.88156941e-01, 6.29477304e-03]

max_sharpe_path = constructPortfolio(weights=weights_max_sharpe,df=df,tickers=tickers,investment=investment)

min_volatility_drag_path = constructPortfolio(weights=sol.x,df=df,tickers=tickers,investment=investment)

# Portfolio Paths
fig, ax = plt.subplots(figsize=(12,8))
fig.suptitle('Portfolio Paths')
ax.plot(max_sharpe_path,label='Maximum Sharpe Path')
ax.plot(min_volatility_drag_path,label='Minimum Volatility Drag Path')
ax.legend()
ax.set_ylabel('Portfolio Value ($)')
ax.set_xlabel('Date')
ax.grid(alpha=0.3)
ax.tick_params('x',rotation=45)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.show()

# Print Statements
print(f'Maximum Sharpe at: {weights_max_sharpe}')
print(f'Minimum Volatility Drag at: {sol.x}')