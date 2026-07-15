# Portfolio Optimisation

![Portfolio Paths](Images/Portfolio_Paths.png)

The first portfolio path (various US Equities) follows an asset weighting aimed at maximising the Sharpe ratio:

$$
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
$$

While the second portfolio aims to minimise the effect of volatility drag and therefore maximise compounding returns:

$$
R_g \approx {\mu} - \frac{1}{2}\sigma^2
$$

It can be seen that although in this case the Sharpe ratio maximised portfolio produces higher returns at the end of the period, the volatility drag optimised portfolio produces more consistent returns when faced with risk, leading to a lower drawdown and long term portfolio convexity. 

Below two optimisation methods are presented, one producing the Efficient Frontier via Monte Carlo simulation, and a second method using an evolutionary algorithm (NSGA2) to produce the Pareto front by simultaneously minimising risk, while maximising expected returns as a multi-objective optimisation problem.

# Efficient Frontier via Monte Carlo Simulation

<p align="center">
  <img src="Images/Monte.png" alt="Logo" width="900">
</p>

# Efficient Frontier via NSGA2 Global Optimisation to find Pareto Front

<p align="center">
  <img src="Images/NSGA2.png" alt="Logo" width="900">
</p>

# Point Minimising Volatility Drag

<p align="center">
  <img src="Images/All.png" alt="Logo" width="900">
</p>
