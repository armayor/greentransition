import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.optimize import minimize

from optimization2 import *



def EstimateParameters(paths):
    estimatedMus = []
    estimatedSigmas = []
    for path in paths:
        logReturns = []
        for k in range (len(path)-1):
            logReturns.append(np.log(path[k+1]/path[k]))
        estimatedSigmas.append(np.std(logReturns)/ np.sqrt(1))
        estimatedMus.append(np.mean(logReturns)/1 + 0.5*np.mean(estimatedSigmas)**2)

    return [np.mean(estimatedMus), np.mean(estimatedSigmas)]









##############################################################################################
################################### INTEREST RATES LOADING ###################################
##############################################################################################

file_path_interest_rates = 'aziende/csv/risk_free_rate_country.csv'
interestRatesFile = pd.read_csv(file_path_interest_rates, delimiter=';', decimal=',')

header = interestRatesFile.columns.tolist()
interestRates = [interestRatesFile[col].to_numpy() for col in interestRatesFile.columns]

# Interest rates (24 values)
rItaly = interestRates[0] 
rFrance = interestRates[1]
rGermany = interestRates[2]

##############################################################################################
########################################## CONVIVO ###########################################
##############################################################################################

file_path_convivo = 'aziende/csv/convivo.csv'
convivoFile = pd.read_csv(file_path_convivo, delimiter=';', decimal=',')

header = convivoFile.columns.tolist()
convivo = [convivoFile[col].to_numpy() for col in convivoFile.columns]

# Interest rates (24 values)
equity = convivo[0][:-1]/1000
Abook = convivo[1][:-1]/100
timeToMaturity = convivo[2][:-1]
K = convivo[3][:-1]/1000

############################## OUTPUT ##############################

N = 24    # Number of time points excluding t=0

# Geometric Brownian Motion Parameters and number of simulations

r = rItaly


result = minimize(lossFunction2, np.array([0.003]), args=(Abook, r, timeToMaturity, np.zeros(N), K, equity,))

print(result)

