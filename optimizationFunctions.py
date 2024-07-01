############################## OUTPUT ##############################

N = 10000    # Number of time points excluding t=0

# Geometric Brownian Motion Parameters and number of simulations
A0 = 100
mu = 0.5
sigma = 0.3742
r = 0.05*np.ones(N)
K = 110

# Time setup
T = 1        # Final time of the simulation
deltat = T/N # Time increment
time = np.linspace(0,T,N+1)


Abook = constructAssetTimeSeries(A0, mu, deltat, sigma, N)

alpha = 1
beta = 7

A = assetValueModifiedFromBook(alpha, beta ,Abook)

callPrices = constructCallPriceTimeSeries(r, A, T, time, sigma, K, N)





#plt.plot(time, A, label = "Asset A value under P")
#plt.plot(time[:-1], callPrices, label = "Call A price")

#plt.legend()
#plt.show()


 


result = minimize(lossFunction, np.array([alpha, beta, sigma])*2, args=(Abook[:-1], r, T, time[:-1], K, callPrices,))
    
print(result)
print()
print(np.array([alpha, beta, sigma])*1.5)
print(np.array([alpha, beta, sigma]))
    


estimatedParameters = EstimateParameters([A])
print("Estimated mu: ", estimatedParameters[0])
print("Estimated sigma: ", estimatedParameters[1])
