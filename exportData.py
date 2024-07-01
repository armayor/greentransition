import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

# Estimates of Parameters from a collection of paths
# @Input: paths
# @Output: vector of estimated [mu, sigma]

def EstimateParameters(paths):
    estimatedMus = []
    estimatedSigmas = []
    for path in paths:
        logReturns = []
        for k in range (N-1):
            logReturns.append(np.log(path[k+1]/path[k]))
        estimatedMus.append(np.mean(logReturns)/deltat + 0.5*sigma**2)
        estimatedSigmas.append(np.std(logReturns)/ np.sqrt(deltat))
    return [np.mean(estimatedMus), np.mean(estimatedSigmas)]


# Black Scholes Pricing of Call Option
# @Input: r: risk-free interest rate, initial
#         A: initial asset price
#         T: maturity
#         t: current time
#     sigma: volatility
#         K: strike price (Book value of liabilities)
#
# @Output: Black Scholes price of a call option

def BlackScholesPriceCall(r, A, T, t, sigma, K):

    d1 = (np.log(A/K)+(r + 0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return norm.cdf(d1)*A - norm.cdf(d2)*K*np.exp(-r*(T-t))


# Loss Function

def lossFunction(parameter, Abook, r, T, t, K, marketCallPrice):
    alpha, beta, sigma = parameter
    A = assetValueModifiedFromBook(alpha, beta, Abook)
    blackScholesPrice = BlackScholesPriceCall(r, A, T, t, sigma, K)

    return np.sum((blackScholesPrice-marketCallPrice)**2)


def assetValueModifiedFromBook(alpha, beta, A):
    return alpha*A + beta



## Construct the time series for the asset price and for the call option

def constructAssetTimeSeries(A0, mu, deltat, sigma, N):
    A = [] # Discretization of the SDE
    A.append(A0) # Initialize A at t = 0 to be A0
    for i in range(N):
        z = np.random.randn()
        A.append(A[i] + mu*A[i]*deltat + sigma*A[i]*np.sqrt(deltat)*z)
    return np.array(A)

def constructCallPriceTimeSeries(r, A, T, time, sigma, K, N):
    callPrices = [] # Call prices at each instant
    callPrices.append(BlackScholesPriceCall(r[0], A[0], T, time[0], sigma, K))

    for i in range(N-1):
        callPrices.append(BlackScholesPriceCall(r[i+1], A[i+1], T, time[i+1], sigma, K))

    return np.array(callPrices)


############################## OUTPUT ##############################

N = 100    # Number of time points excluding t=0

# Geometric Brownian Motion Parameters and number of simulations
A0 = 100
mu = 0.5
sigma = 0.26
r = 0.05*np.ones(N)
K = 110

# Time setup
T = 1        # Final time of the simulation
deltat = T/N # Time increment
time = np.linspace(0,T,N+1)


Abook = constructAssetTimeSeries(A0, mu, deltat, sigma, N)
A = assetValueModifiedFromBook(2,1,Abook)
callPrices = constructCallPriceTimeSeries(r, A, T, time, sigma, K, N)

dati = np.array([Abook[:-1], A[:-1], callPrices, T-time[:-1]]).T
pd.DataFrame(data=dati, columns=['Abook','A','Equity','Mat']).to_excel('simulated_model.xlsx')
print(dati)