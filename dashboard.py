import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
import os

plt.style.use('dark_background')

end = dt.datetime.now()
start = end - dt.timedelta(days=252)
interval = '1d'
ticker = 'TSLA'
market_ticker = 'SPY'
window = 21
PERIODS_PER_YEAR = 252

data_directory = r'C:\Users\migue\Desktop\Part IV\Quant\Fama French Data'

fama_french3f_filename = 'F-F Data 3 Factor.csv'
fama_french5f_filename = 'F-F Data 5 Factor.csv'


# ---------------------------------------------------------------------------
# Function definitions and helpers
# ---------------------------------------------------------------------------

def example_data_download(tickers,start,end,interval) -> pd.Series:
    data = yf.download(tickers=tickers,start=start,end=end,interval=interval)
    data = data['Close'].squeeze()
    data.to_csv("Example_Data.csv",index=True)
    return data

# data = example_data_download(tickers=ticker,start=start,end=end,interval=interval)

def compute_returns(df: pd.Series):
    returns = df.pct_change().dropna()
    return returns

def wealth_index(returns: pd.Series) -> pd.Series:
    initial = 1.0
    wealth = initial * (1.0 + returns).cumprod()
    start_date = wealth.index.min() - pd.DateOffset(days=1)
    wealth = pd.concat([pd.Series([1],index=[start_date]),wealth]) # Thanks to Daniel Boctor
    return wealth

def compute_annualised_volatility(returns: pd.Series,PERIODS_PER_YEAR = 252):
    vol = returns.std()*np.sqrt(PERIODS_PER_YEAR)
    vol *= 100
    return vol

def compute_annualised_return(wealth: pd.Series, PERIODS_PER_YEAR = 252):
    cagr = (wealth.iloc[-1] / wealth.iloc[0]) ** (252 / PERIODS_PER_YEAR) - 1
    cagr = cagr.item()
    return cagr

def compute_annualised_sharpe(returns: pd.Series, rf_daily,PERIODS_PER_YEAR = 252):
    excess = returns - rf_daily
    vol = returns.std()
    sharpe = excess.mean() / vol * np.sqrt(PERIODS_PER_YEAR)
    return sharpe.item()

def compute_annualised_sortino(returns: pd.Series,rf_daily,PERIODS_PER_YEAR = 252):
    excess = returns - rf_daily
    downside = excess.clip(upper=0)
    downside_vol = np.sqrt(np.mean(downside**2))
    sortino = (excess.mean() / downside_vol) * np.sqrt(PERIODS_PER_YEAR)
    return sortino.item()

def compute_drawdowns(wealth: pd.Series):
    previous_peaks = wealth.cummax()
    drawdowns = (wealth - previous_peaks)/previous_peaks
    return drawdowns

def compute_max_drawdown(wealth: pd.Series):
    previous_peaks = wealth.cummax()
    max_drawdown = ((wealth - previous_peaks) / previous_peaks).min().item() * 100
    return max_drawdown
 
def benchmark(tickers,start,end,interval) -> pd.Series:
    benchmark = yf.download(tickers=tickers,start=start,end=end,interval=interval)
    market = benchmark['Close']
    return market

def compute_rolling_sharpe(returns: pd.Series,window, rf_daily,PERIODS_PER_YEAR = 252) -> pd.Series:
    excess = returns - rf_daily
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std()
    sharpe = (mean / std) * np.sqrt(PERIODS_PER_YEAR)
    return sharpe

def compute_rolling_sortino(returns: pd.Series,window, rf_daily,PERIODS_PER_YEAR = 252) -> pd.Series:
    excess = returns - rf_daily
    downside = excess.clip(upper=0)
    mean = excess.rolling(window).mean()
    downside_vol = np.sqrt((downside ** 2).rolling(window).mean())
    sortino = (mean / downside_vol) * np.sqrt(PERIODS_PER_YEAR)
    return sortino

def capm_regression(returns, market, rf_daily,PERIODS_PER_YEAR = 252):
    market_returns = market.pct_change()

    df = pd.concat([returns - rf_daily,market_returns - rf_daily],axis=1).dropna()
    df.columns = ['asset_excess', 'market_excess']

    X = sm.add_constant(df['market_excess'])
    y = df['asset_excess']

    model = sm.OLS(y, X).fit()

    alpha = model.params['const'] * PERIODS_PER_YEAR
    beta = model.params['market_excess']

    line = df['market_excess'] * beta + alpha / PERIODS_PER_YEAR

    r_squared = model.rsquared

    print(model.summary())

    metrics = pd.DataFrame({"Coefficient": round(model.params,4),
    "Std Error": round(model.bse,4),
    "t-stat": round(model.tvalues,4),
    "p-value": round(model.pvalues,4)})

    return alpha, beta, df, line, r_squared, metrics

def import_data(filename,directory = data_directory):
    path = os.path.join(directory,filename)
    ff_data = pd.read_csv(path)

    # Format dates
    dates = []
    for date in ff_data['Date'].values:
        formatted_date = dt.datetime.strptime(f"{date}", "%Y%m%d")
        dates.append(formatted_date)

    ff_data = ff_data.set_index(pd.Series(dates))

    return ff_data

def three_factor_model(returns,ff_data,PERIODS_PER_YEAR = 252):
    # Trim FF data
    ff_data[['RF', 'Mkt-RF', 'SMB', 'HML']] /= 100
    ff_data = ff_data.loc[ff_data.index.isin(returns.index)]

    # Regression
    rf_daily = ff_data['RF']
    df = pd.concat([returns - rf_daily,ff_data['Mkt-RF'],ff_data['SMB'],ff_data['HML']],axis=1).dropna()
    df.columns = ['asset_excess','market_excess','smb','hml']

    X = df[['market_excess','smb','hml']]
    X = sm.add_constant(X)
    y = df['asset_excess']

    model = sm.OLS(y,X).fit()

    alpha = model.params['const'] * PERIODS_PER_YEAR
    beta1 = model.params['market_excess']
    beta2 = model.params['smb']
    beta3 = model.params['hml']

    results = model.summary()
    print(results)

    metrics = pd.DataFrame({"Coefficient": round(model.params,4),
        "Std Error": round(model.bse,4),
        "t-stat": round(model.tvalues,4),
        "p-value": round(model.pvalues,4)})

    r_squared = model.rsquared

    adj = model.rsquared_adj

    return alpha,beta1,beta2,beta3,metrics,r_squared,adj

def five_factor_model(returns,ff_data,PERIODS_PER_YEAR = 252):
    # Trim FF data
    ff_data = ff_data.loc[ff_data.index.isin(returns.index)]
    ff_data[['RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] /= 100

    # Regression
    rf_daily = ff_data['RF']
    df = pd.concat([returns - rf_daily,ff_data['Mkt-RF'],ff_data['SMB'],ff_data['HML'],ff_data['RMW'],ff_data['CMA']],axis=1).dropna()
    df.columns = ['asset_excess','market_excess','smb','hml','rmw','cma']

    X = df[['market_excess','smb','hml','rmw','cma']]
    X = sm.add_constant(X)
    y = df['asset_excess']

    model = sm.OLS(y,X).fit()

    alpha = model.params['const'] * PERIODS_PER_YEAR
    beta1 = model.params['market_excess']
    beta2 = model.params['smb']
    beta3 = model.params['hml']
    beta4 = model.params['rmw']
    beta5 = model.params['cma']

    results = model.summary()
    print(results)

    metrics = pd.DataFrame({"Coefficient": round(model.params,4),
        "Std Error": round(model.bse,4),
        "t-stat": round(model.tvalues,4),
        "p-value": round(model.pvalues,4)})
    
    r_squared = model.rsquared

    adj = model.rsquared_adj

    return alpha,beta1,beta2,beta3,beta4,beta5,metrics,r_squared,adj

def rolling_three_factor(returns,ff_data,window):
    ff_data[['RF', 'Mkt-RF', 'SMB', 'HML']] /= 100
    df = pd.concat([returns,ff_data[['RF', 'Mkt-RF', 'SMB', 'HML']]], axis=1).dropna()
    df.columns = ['returns','RF','market_excess','smb','hml']
    rolling_results = []

    for i in range(window, len(df) + 1):
        window_df = df.iloc[i-window:i]
        y = window_df['returns'] - window_df['RF']
        
        X = window_df[['market_excess', 'smb', 'hml']]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        rolling_results.append({'date': window_df.index[-1],
        'beta1': model.params['market_excess'],'beta2': model.params['smb'],
        'beta3': model.params['hml']})

    results = pd.DataFrame(rolling_results)
    results = results.set_index('date')

    return results

def rolling_five_factor(returns,ff_data,window):
    ff_data[['RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] /= 100
    df = pd.concat([returns,ff_data[['RF', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]], axis=1).dropna()
    df.columns = ['returns','RF','market_excess','smb','hml','rmw','cma']
    rolling_results = []

    for i in range(window, len(df) + 1):
        window_df = df.iloc[i-window:i]
        y = window_df['returns'] - window_df['RF']
        
        X = window_df[['market_excess', 'smb', 'hml','rmw','cma']]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        rolling_results.append({'date': window_df.index[-1],
        'beta1': model.params['market_excess'],'beta2': model.params['smb'],
        'beta3': model.params['hml'], 'beta4': model.params['rmw'],
        'beta5': model.params['cma']})

    results = pd.DataFrame(rolling_results)
    results = results.set_index('date')

    return results

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Quantitative Metrics Sheet",text_alignment='center')

st.text("Upload file or input ticker for metrics:",text_alignment='center')
st.markdown("* Start date (integer past days or date in Y-m-d).",text_alignment='left')
st.markdown("* End date (NOW or date in Y-m-d).",text_alignment='left')
st.markdown("* Interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). *Note intraday data not available for period greater than 60 days).",text_alignment='left')

uploaded_file = st.file_uploader("Upload data",type="csv")

col1, col2, col3, col4 = st.columns(4)

with col1:
    ticker_input = st.text_input("Ticker")

with col2:
    start_input = st.text_input("Start Date")

with col3:
    end_input = st.text_input("End Date")

with col4:
    interval_input = st.text_input("Interval")

option = st.selectbox("Window Size",("21", "42", "63","126","252"))

rf_annual = 0.04
rf_daily = (1 + rf_annual)**(1/252) - 1

if option:
    window = int(option)
else:
    window = 21

data = None

# Data calls
if uploaded_file is not None:
    data_file = pd.read_csv(uploaded_file,parse_dates=['Date'])

    data = data_file.set_index('Date')

    data = data.iloc[:,0].squeeze()

    start = data.index[0]

    end = data.index[-1]

    interval = '1d'

elif all([ticker_input, start_input, end_input, interval_input]):
    ticker = ticker_input.upper()
    if (end_input.upper() == "NOW") and (start_input.isdigit()):
        end = dt.datetime.now()
        start = end - dt.timedelta(days=int(start_input))

    elif (isinstance(start_input, str)) and (isinstance(end_input, str)) and (end_input.upper() != "NOW"):
        start = dt.datetime.strptime(start_input, "%Y-%m-%d")
        end = dt.datetime.strptime(end_input, "%Y-%m-%d")

    elif (isinstance(start_input, str)) and (end_input.upper() == "NOW"):
        start = dt.datetime.strptime(start_input, "%Y-%m-%d")
        end = dt.datetime.now()

    elif (start_input.isdigit()) and (isinstance(end_input, str)):
        end = dt.datetime.strptime(end_input, "%Y-%m-%d")
        start = end - dt.timedelta(days=int(start_input))
        
    interval = interval_input

    data = yf.download(tickers=ticker,start=start,end=end,interval=interval)

    data = data['Close'].squeeze()
    
if data is not None:
    market = yf.download(tickers=market_ticker,start=start,end=end,interval=interval)

    data = data.truncate(after=market.index[-1])

    market = market['Close'].squeeze()

    returns = compute_returns(df=data)

    wealth = wealth_index(returns=returns)

    cagr = compute_annualised_return(wealth=wealth)

    alpha, beta, df, line, r_squared, metrics_capm = capm_regression(returns=returns,rf_daily=rf_daily,market=market)

    sharpe = compute_annualised_sharpe(returns=returns,rf_daily=rf_daily)

    sortino = compute_annualised_sortino(returns=returns,rf_daily=rf_daily)

    drawdowns = compute_drawdowns(wealth=wealth)

    max_drawdown = compute_max_drawdown(wealth=wealth)

    market_returns = compute_returns(df=market)

    wealth_index_market = wealth_index(returns=market_returns)

    vol = compute_annualised_volatility(returns=returns)

    rolling_sharpe = compute_rolling_sharpe(returns=returns,window=window,rf_daily=rf_daily)

    rolling_sortino = compute_rolling_sortino(returns=returns,window=window,rf_daily=rf_daily)

    # Fama-French 3 Factor Model
    ff_data3f = import_data(filename = fama_french3f_filename)
    alpha_3f,beta1_3f,beta2_3f,beta3_3f,metrics_3f,r_squared_3f,adj_3f = three_factor_model(returns=returns,ff_data=ff_data3f)
    results_3f = rolling_three_factor(returns=returns,ff_data=ff_data3f,window=window)

    # Fama-French 5 Factor Model
    ff_data5f = import_data(filename = fama_french5f_filename)
    alpha_5f,beta1_5f,beta2_5f,beta3_5f,beta4_5f,beta5_5f,metrics_5f,r_squared_5f,adj_5f = five_factor_model(returns=returns,ff_data=ff_data5f)
    results_5f = rolling_five_factor(returns=returns,ff_data=ff_data5f,window=window)

    col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Alpha", f"{alpha:.2f}","CAPM Regression Intercept",border=True)
    # col2.metric("Beta", f"{beta:.2f}","CAPM Regression Coefficient",border=True)
    # col3.metric("Sharpe", f"{sharpe:.2f}","Annualised",border=True)
    # col4.metric("Sortino",f"{sortino:.2f}","Annualised",border=True)
    col1.metric("Annualised Alpha", f"{alpha:.2f}",border=True)
    col2.metric("Beta", f"{beta:.2f}",border=True)
    col3.metric("Annualised Sharpe", f"{sharpe:.2f}",border=True)
    col4.metric("Annualised Sortino",f"{sortino:.2f}",border=True)

    col5, col6, col7, col8 = st.columns(4)
    # col5.metric("CAGR", f"{cagr:.2f}","Compount Annual Growth Rate",border=True)
    # col6.metric("Annualised Volatility", f"{vol:.2f}%","Annualied",border=True)
    # col7.metric("Max Drawdown", f"{max_drawdown:.2f}%","Largest Loss",border=True)
    # col8.metric("R² CAPM",f"{r_squared:.2f}","CAPM Regression Market Variance Explained",border=True)
    col5.metric("CAGR", f"{cagr:.2f}",border=True)
    col6.metric("Annualised Volatility", f"{vol:.2f}%",border=True)
    col7.metric("Max Drawdown", f"{max_drawdown:.2f}%",border=True)
    col8.metric("CAPM R²",f"{r_squared:.2f}",border=True)

    # Wealth index plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wealth.index, 
            y=wealth.values, 
            name=f"Portfolio", 
            mode="lines",
            line=dict(color="#4ade80", width=2.2),
            fill="tozeroy", 
            fillcolor="rgba(74,222,128,0.08)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=wealth_index_market.index, 
            y=wealth_index_market.values, 
            name="SPY", 
            mode="lines",
            line=dict(color="#60a5fa", width=1.8, dash="dot")
        )
    )

    fig.update_yaxes(tickprefix="$")

    fig.update_layout(
        title="Wealth Index",
        xaxis_title="Date",
        yaxis_title="Equity Curve on $1",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    # Drawdowns and rolling sharpe and sortino
    fig = make_subplots(rows=1,cols=2,subplot_titles=("Underwater Plot", "Rolling Risk-Adjusted Returns"))

    fig.add_trace(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values*100,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            fillcolor="rgba(248,113,113,0.20)",
            line=dict(color="#f87171", width=1.4)
        ),
        row=1, 
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Sharpe Ratio",
            fill='tozeroy',
            line=dict(color="rgba(251,191,36,0.08)", width=1.4)
        ),
        row=1, 
        col=2
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_sortino.index,
            y=rolling_sortino.values,
            mode="lines",
            name="Sortino Ratio",
            fill='tozeroy',
            line=dict(color="#60a5fa", width=1.4)
        ),
        row=1, 
        col=2
    )

    fig.update_yaxes(
        ticksuffix="%",
        title_text="Drawdown",
        row=1,
        col=1
    )

    fig.update_yaxes(ticksuffix="%",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Ratio",
        row=1,
        col=2
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    # CAPM regression plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['market_excess'].squeeze(), 
            y=df['asset_excess'].squeeze(), 
            name="Returns", 
            mode="markers",
            line=dict(color="#4ade80"),
            fill="tozeroy", 
            fillcolor="rgba(74,222,128,0.08)"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['market_excess'].squeeze(), 
            y=line.values, 
            name="CAPM Fit", 
            mode="lines",
            line=dict(color="#f87171", width=1.8)
        )
    )

    fig.update_layout(
        title=f"CAPM Regression: R² = {r_squared:.2f}",
        xaxis_title="Market Returns (Excess)",
        yaxis_title="Asset Returns (Excess)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    # Factor loadings
    df_3f = pd.DataFrame({'Beta':[beta1_3f,beta2_3f,beta3_3f]},index=['Mkt-RF','SMB','HML'])

    df_5f = pd.DataFrame({'Beta':[beta1_5f,beta2_5f,beta3_5f,beta4_5f,beta5_5f]},index=['Mkt-RF','SMB','HML','RMW','CMA'])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Fama-French 5 Factor Loadings: R² = {r_squared_5f:.2f}",f"Fama-French 3 Factor Loadings: R² = {r_squared_3f:.2f}")
    )

    fig.add_trace(
        go.Bar(
            x=df_3f.index,
            y=df_3f['Beta'],
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Bar(
            x=df_5f.index,
            y=df_5f['Beta'],
            showlegend=False
        ),
        row=1,
        col=1
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    # Fama French rolling factors
    fig = go.Figure()

    fig.update_layout(
        title=f"Fama French Rolling 5 Factor Betas ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark"
    )

    fig.add_trace(
        go.Scatter(
            x=results_5f.index,
            y=results_5f['beta1'].values,
            mode="lines",
            name="Mkt-RF",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_5f.index,
            y=results_5f['beta2'].values,
            mode="lines",
            name="SMB",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_5f.index,
            y=results_5f['beta3'].values,
            mode="lines",
            name="HML",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_5f.index,
            y=results_5f['beta4'].values,
            mode="lines",
            name="RMW",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_5f.index,
            y=results_5f['beta5'].values,
            mode="lines",
            name="CMA",
            line=dict(width=1.8)
        )
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    fig = go.Figure()

    fig.update_layout(
        title=f"Fama French Rolling 3 Factor Betas ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark"
    )

    fig.add_trace(
        go.Scatter(
            x=results_3f.index,
            y=results_3f['beta1'].values,
            mode="lines",
            name="Mkt-RF",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_3f.index,
            y=results_3f['beta2'].values,
            mode="lines",
            name="SMB",
            line=dict(width=1.8)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_3f.index,
            y=results_3f['beta3'].values,
            mode="lines",
            name="HML",
            line=dict(width=1.8)
        )
    )

    st.plotly_chart(fig, use_container_width=True, config = {'scrollZoom': False})

    # Model statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"5 Factor Regression: R² = {r_squared_5f:.2f}, Adjusted R² = {adj_5f:.2f}")
        st.dataframe(
            metrics_5f.style.format("{:.4f}"),
            use_container_width=True,
            hide_index=False
        )

    metrics_3f = pd.concat([metrics_3f,metrics_capm],axis=0)
    new_index = ['const','market_excess','smb','hml','alpha (CAPM)','beta (CAPM)']
    metrics_3f.index = new_index

    with col2:
        st.subheader(f"3 Factor Regression: R² = {r_squared_3f:.2f}, Adjusted R² = {adj_3f:.2f}")
        st.dataframe(
            metrics_3f.style.format("{:.4f}"),
            use_container_width=True,
            hide_index=False
        )