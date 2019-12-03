import numpy as np
import matplotlib.pyplot as plt
import TempSim
import networkx as nx
#%matplotlib qt

def FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents):
    fitness, maxFitness = TempSim.runSimulation(voiPositions, nCities, nAgents, cityMap, cityPositions, agents)
    return fitness, maxFitness

def initAgents(nAgents, nNodes):
    agents = np.zeros((nAgents, 3), dtype=np.int8)
    for i in range(nAgents):
        cityIndexes = [x for x in range(nNodes)]
        startCity = np.random.choice(cityIndexes, 1)
        currentCity = startCity
        cityIndexes.remove(startCity)
        endCity = np.random.choice(cityIndexes, 1)
        agents[i, 0] = currentCity
        agents[i, 1] = startCity
        agents[i, 2] = endCity
    return agents

'''
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
'''
    
plt.close("all")

#Import map to use
data_set = np.load('MapToUse.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
nCities = np.size(cityMap,0)

#Model parameters
nAgents = 100
nVois = 1*nCities
nTimeSteps = 100

agents = initAgents(nAgents, nCities)

voiPositions = np.ones(nCities)*nVois/nCities

fitness = np.zeros(nTimeSteps)
maxFitness = np.zeros(nTimeSteps)
for iTime in range(nTimeSteps):
    if np.mod(iTime+1, nTimeSteps/10) == 0:
        print('Progress: ' + str((iTime+1)/nTimeSteps*100) + ' %')
        
    voiPositions = np.ones(nCities)*nVois/nCities  
    fitness[iTime], maxFitness[iTime] = FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents)  
    
plt.figure()
plt.plot(fitness,'r')
plt.plot(maxFitness,'--k')

plt.figure()
G = nx.from_numpy_matrix(cityMap, create_using=nx.DiGraph())
poss = {}
for i in range(nCities):
    poss[i] = cityPositions[i]
labels = {}
for i in range(nCities):
    labels[i] = voiPositions[i]
#nx.draw_networkx(G,poss)
nx.draw_networkx(G, poss, labels=labels)
plt.show()