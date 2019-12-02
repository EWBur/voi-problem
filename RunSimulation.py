import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt

def FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions):
    fitness, maxFitness = TempSim.runSimulation(voiPositions, nCities, nAgents, cityMap, cityPositions)
    return fitness, maxFitness

def PlotGraph(edges, nodes):
    indecesOfEdges = np.where(edges != 0)
    fromCityPositions = [nodes[indecesOfEdges[0], 0],
                         nodes[indecesOfEdges[0], 1]]
    toCityPositions = [nodes[indecesOfEdges[1], 0],
                       nodes[indecesOfEdges[1], 1]]

    plt.figure()
    ax = plt.gca().set_aspect('equal', adjustable='box')
    markerSize = 10
    fontSize = 30
    lineWidth = 1
    plt.xlabel('x', fontsize=fontSize)
    plt.ylabel('y', fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)

    plt.plot([fromCityPositions[0][:], toCityPositions[0][:]], [
             fromCityPositions[1][:], toCityPositions[1][:]], 'k', linewidth=lineWidth)
    plt.plot(nodes[:, 0], nodes[:, 1], 'or', markersize=markerSize)
    plt.show()
    
plt.close("all")

#Model parameters
nAgents = 100
nVois = 3*nCities
nTimeSteps = 100

data_set = np.load('MapToUse.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
nCities = np.size(cityMap,0)

PlotGraph(cityMap, cityPositions)

voiPositions = np.ones(nCities)*nVois/nCities

fitness = np.zeros(nTimeSteps)
maxFitness = np.zeros(nTimeSteps)
for iTime in range(nTimeSteps):
    #voiPositions = np.ones(nCities)*nVois/nCities
  
    fitness[iTime], maxFitness[iTime] = FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions)
    
print(voiPositions)    
    
plt.figure()
plt.plot(fitness,'r')
plt.plot(maxFitness,'--k')
plt.show()