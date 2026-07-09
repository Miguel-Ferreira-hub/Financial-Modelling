# Simulation of various stochastic processes and models
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# Arithmetic Brownian Motion
def ABM(mu, sigma, S0, n_steps):
    T = 1
    dt = 1 / n_steps
    t = np.linspace(dt, T, n_steps)
    S = np.zeros(len(t))
    S[0] = S0
    # Euler-Maruyama Discretisation (same as closed form)
    for i in range(1,len(S)):
        S[i] = S[i-1] + mu*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)
    return t, S

# Geometric Brownian Motion - degenerate solution as S0 approaches 0 or sigma = 0
def GBM(mu, sigma, S0, n_steps):
    T = 1
    dt = T / n_steps # Change in step
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))  # Wiener Process (Standard Brownian Motion)
    t = np.linspace(dt, T, n_steps) # Timestep domain
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W) # Geometric Brownian Motion - Exact Solution (closed form)
    return t, S

# Ornstein-Uhlenbeck Process (Mean Reverting Geometric Brownian Motion)
def OU(mu, sigma, theta, S0, n_steps):
    T = 1
    dt = T / n_steps
    t = np.linspace(dt,T,n_steps)
    X = np.zeros(len(t))
    X[0] = S0
    a = np.exp(-theta*dt)
    b = np.sqrt((sigma**2/(2*theta))*(1-np.exp(-2*theta*dt)))
    for i in range(1,len(t)):
        X[i] = mu + a*(X[i-1] - mu) + b*np.random.normal(0,1)
    return t, X

# Merton Jump Diffusion Model
def MJD(mu, sigma, jump, jump_rate, S0, n_steps):
    T = 1
    dt = T / n_steps
    t = np.linspace(dt,T,n_steps)
    S = np.zeros(len(t))
    S[0] = S0
    # Euler-Maruyama Discretisation
    for i in range(1,len(t)):
        S[i] = S[i-1] + mu*dt*S[i-1] + sigma*S[i-1]*np.sqrt(dt)*np.random.normal(0,1) + S[i-1]*(jump-1)*np.random.poisson(jump_rate*dt)
    return t, S

# Heston Stochastic Volatility Model
def Heston(mu, theta, zeta, kappa, rho, v0, S0, n_steps):
    T = 1
    dt = T/ n_steps
    t = np.linspace(dt,T,n_steps)
    S = np.zeros(len(t))
    v = np.zeros(len(t))
    v[0] = v0
    S[0] = S0
    # Euler-Maruyama Discretisation
    for i in range(1,len(t)):
        Z1 = np.random.normal(0,1)
        Z2 = np.random.normal(0,1)
        Zv = Z1
        Zs = rho*Z1 + np.sqrt(1-rho**2)*Z2
        v[i] = v[i-1] + kappa*(theta - v[i-1])*dt + zeta*np.sqrt(v[i-1])*np.sqrt(dt)*Zv
        S[i] = S[i-1] + mu*S[i-1]*dt + np.sqrt(v[i-1])*S[i-1]*np.sqrt(dt)*Zs
    return t, S

# Initial Parameters
mu = 1
sigma = 0.4
S0 = 1
theta = 15
jump = 0.5 # Constant Jump Size (Simplification)
jump_rate = 0.20
kappa = 3
zeta = 0.5
rho = -0.5 # Typically between -0.9 to -0.3
v0 = 0.4
n_steps = 1000

t0, path0 = ABM(mu=mu, sigma=sigma, S0=S0, n_steps=n_steps) # Arithmetic Brownian Motion

t1, path1 = GBM(mu=mu, sigma=sigma, S0=S0, n_steps=n_steps) # Geometric Brownian Motion

t2, path2 = OU(mu=mu, sigma=sigma, theta=theta, S0=S0, n_steps=n_steps) # Ornsetein-Uhlenbeck Process

t3, path3 = MJD(mu=mu, sigma=sigma, jump=jump, jump_rate=jump_rate, S0=S0, n_steps=n_steps) # Merton Jump Diffusion

t4, path4 = Heston(mu=mu, theta=theta, zeta=zeta, kappa=kappa, rho=rho, v0=v0, S0=S0, n_steps=n_steps) # Heston 

# Plot
plt.figure(figsize=(12,10))
plt.title("Stochastic Processes")
plt.plot(t0,path0,label='Arithmetic Brownian Motion')
plt.plot(t1,path1,label='Geometric Brownian Motion')
plt.plot(t2,path2,label='Ornstein-Uhlenbeck Process')
plt.plot(t3,path3,label='Merton Jump Diffusion Model')
plt.plot(t4,path4,label='Heston Stochastic Volatility Model')
plt.xlabel('Time')
plt.ylabel('S(t)')
plt.hlines(mu,xmin=0,xmax=1,color='red',label='OU Mean')
plt.legend()
plt.grid(alpha=0.7)
plt.show()