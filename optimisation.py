import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.optimize import minimize


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

def lossFunction(parameter, Abook, sigma, r, T, t, K, marketCallPrice):
    alpha, beta = parameter
    A = assetValueModifiedFromBook(alpha, beta, Abook)
    blackScholesPrice = BlackScholesPriceCall(r, A, T, t, sigma, K)

    return (blackScholesPrice-marketCallPrice)**2


def assetValueModifiedFromBook(alpha, beta, A):
    return alpha*A + beta



## Construct the time series for the asset price and for the call option

def constructAssetTimeSeries(A0, mu, deltat, sigma, N):
    A = [] # Discretization of the SDE
    A.append(A0) # Initialize A at t = 0 to be A0
    for i in range(N):
        z = np.random.randn()
        A.append(A[i] + mu*A[i]*deltat + sigma*A[i]*np.sqrt(deltat)*z)
    return A

def constructCallPriceTimeSeries(r, A, T, time, sigma, K, N):
    callPrices = [] # Call prices at each instant
    callPrices.append(BlackScholesPriceCall(r, A[0], T, time[0], sigma, K))

    for i in range(N):
        callPrices.append(BlackScholesPriceCall(r, A[i+1], T, time[i+1], sigma, K))

    return callPrices


############################## OUTPUT ##############################

# Geometric Brownian Motion Parameters and number of simulations
A0 = 100
mu = 0.5
sigma = 0.26
r = 0.05
K = 110

# Time setup
T = 1        # Final time of the simulation
N = 10    # Number of time points excluding t=0
deltat = T/N # Time increment
time = np.linspace(0,T,N+1)


Abook = np.array(constructAssetTimeSeries(A0, mu, deltat, sigma, N))
A = assetValueModifiedFromBook(2,1,Abook)
callPrices = np.array(constructCallPriceTimeSeries(r, A, T, time, sigma, K, N))


print(A)

estimatedAssetValues = []

for i in range(N):
    result = minimize(lossFunction, [300], args=(sigma, r, T, time[i], K, callPrices[i],))
    estimatedAssetValues.append(result)


    


plt.plot(time, A, label = "Asset A value under P")
#plt.plot(time[:-1], estimatedAssetValues, label = "Estimated Asset A value")
plt.plot(time, callPrices, label = "Call A price")

plt.legend()
plt.show()

estimatedParameters = EstimateParameters([A])
print("Estimated mu: ", estimatedParameters[0])
print("Estimated sigma: ", estimatedParameters[1])