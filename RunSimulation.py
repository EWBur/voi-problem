import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt

def FitnessOfPopulation(voiPositions, nCities, nAgents):
    fitness, maxFitness = TempSim.runSimulation(voiPositions, nCities, nAgents)
    return fitness, maxFitness

#Model parameters
nCities = 20
nAgents = 10
nVois = 5*nCities
nTimeSteps = 100

data_set = np.load('Test2.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']

voiPositions = np.ones(nCities)*nVois/nCities

fitness = np.zeros(nTimeSteps)
maxFitness = np.zeros(nTimeSteps)
for iTime in range(nTimeSteps):
    #voiPositions = np.ones(nCities)*nVois/nCities
  
    fitness[iTime], maxFitness[iTime] = FitnessOfPopulation(voiPositions, nCities, nAgents)
    
print(voiPositions)    
    
plt.figure()
plt.plot(fitness,'r')
plt.plot(maxFitness,'--k')
plt.show()