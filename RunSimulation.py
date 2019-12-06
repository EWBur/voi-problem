import numpy as np
import matplotlib.pyplot as plt
import TempSim
import networkx as nx
#%matplotlib qt

def FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents):
    fitness, maxFitness = TempSim.runSimulation(voiPositions, nCities, nAgents, cityMap, cityPositions, agents, nGroups, mutationProbabilityAgents)
    return fitness, maxFitness

'''
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

def FindGraphCenter(nodePositions):
    networkCenter = np.sum(nodePositions,0)/np.size(nodePositions,0)
    return networkCenter

def VoiDistanceFromCenter(nodePositions,voiPositions,networkCenter):
    voiDistanceFromCenter = np.matmul(voiPositions,np.sqrt(np.sum((nodePositions-networkCenter)**2,1)))/np.sum(voiPositions)
    return voiDistanceFromCenter

def PlotGraphAndVois(cityMap,nCities,voiPositions,cityPositions):
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
    
def PlotFitness(fitness,maxFitness):
    fontSize = 20
    
    plt.figure()
    plt.plot(fitness,'r')
    plt.plot(maxFitness,'--k')
    
    plt.xlabel('Time',fontsize=fontSize)
    plt.ylabel('Scooter usage',fontsize=fontSize)
    plt.legend(['Actual usage','Maximum usage'],fontsize=fontSize,frameon=False)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('Scooter usage',fontsize=fontSize)
    
    plt.figure()
    plt.plot(maxFitness-fitness,'k')
    
    plt.xlabel('Time',fontsize=fontSize)
    plt.ylabel('Voi usage',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('Difference of maximum voi usage and actual voi usage',fontsize=fontSize)
    
def PlotVoiDistanceFromCenter(voiDistanceFromCenter):
    fontSize = 20
    
    plt.figure()
    plt.plot(voiDistanceFromCenter,'k')
    
    plt.xlabel('Time',fontsize=fontSize)
    plt.ylabel('Mean distance',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('The scooters mean distance from the networks center',fontsize=fontSize)
    
def PlotAgentsStartEndDistribution(agents,nNodes):
    fontSize = 20
    
    plt.figure()
    plt.hist(agents[:,1],nNodes,fc=(0, 0, 1, 0.5))
    plt.hist(agents[:,2],nNodes,fc=(1, 0, 0, 0.5))
    
    plt.xlabel('Node index',fontsize=fontSize)
    plt.ylabel('Number of agents',fontsize=fontSize)
    plt.legend(['Start node','End node'],fontsize=fontSize,frameon=False)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('Distribution of agents start and end',fontsize=fontSize)
    
plt.close("all")

#Import map to use and agents
data_set = np.load('MapToUseNew.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
uniformAgents = data_set['uniformAgents']
nCities = np.size(cityMap,0)

#Compute the graphs center
networkCenter = FindGraphCenter(cityPositions)

#Model parameters
nAgents = 100
nVois = 2*nCities
nTimeSteps = 5
nGroups = nAgents
mutationProbabilityAgents = 0

#Load agents
agents = np.zeros((nAgents,3),int)
agents[0:nAgents,:] = uniformAgents[0:nAgents,:]

#Initial voi distribution (uniform)
voiPositions = np.ones(nCities)*nVois/nCities

fitness = np.zeros(nTimeSteps)
maxFitness = np.zeros(nTimeSteps)
voiDistanceFromCenter = np.zeros(nTimeSteps)

PlotAgentsStartEndDistribution(agents,nCities)

for iTime in range(nTimeSteps):
    if np.mod(iTime+1, nTimeSteps/10) == 0:
        print('Progress: ' + str((iTime+1)/nTimeSteps*100) + ' %')
        
    #voiPositions = np.ones(nCities)*nVois/nCities ### RESETS ALL VOI POSITIONS EVERY DAY (UNIFORMLY)
     
    agents[0:nAgents,:] = uniformAgents[0:nAgents,:]
    fitness[iTime], maxFitness[iTime] = FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents)  
    
    voiDistanceFromCenter[iTime] = VoiDistanceFromCenter(cityPositions,voiPositions,networkCenter)

#Plots
PlotGraphAndVois(cityMap,nCities,voiPositions,cityPositions)
PlotFitness(fitness,maxFitness)
PlotVoiDistanceFromCenter(voiDistanceFromCenter)
plt.show()