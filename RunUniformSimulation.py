import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt

def FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents):
    fitness, maxFitness, newVoiPositions, nodeUsage = TempSim.runSimulation(voiPositions, nCities, nAgents, cityMap, cityPositions, agents, nGroups, mutationProbabilityAgents)
    return fitness, maxFitness, newVoiPositions, nodeUsage

def PlotFitness(fitness,maxFitness,nAgents,nVois,nCities):
    fontSize = 20
    markerSize = 15
    
    plt.figure()
    for iAgents in range(len(nAgents)):
        plt.plot(nVois/nCities,np.divide(fitness[:,iAgents],maxFitness[:,iAgents]),'o',markersize=markerSize)
    
    plt.xlabel('Number of vois per node',fontsize=fontSize)
    plt.ylabel('Relative voi usage',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('Actual voi usage relative to maximum voi usage',fontsize=fontSize)
    plt.legend(nAgents,fontsize=fontSize,frameon=False,title='Number of agents', title_fontsize=fontSize)
    
plt.close("all")

#Import map to use and agents
data_set = np.load('MapToUse4.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
uniformAgents = data_set['uniformAgents']
distributedAgents = data_set['distributedAgents']
nCities = np.size(cityMap,0)

#Model parameters
nAgents = np.asarray([50,100,150,200,250,300,400,500])
nVois = np.asarray([1,2,3])*nCities
nRepetitions = 1

mutationProbabilityAgents = 0
nGroupsPartial = 1

fitnessData = np.zeros((len(nVois),len(nAgents)))
maxFitnessData = np.zeros((len(nVois),len(nAgents)))

for iVois in range(len(nVois)):
    for jAgents in range(len(nAgents)):
        for kRepetitions in range(nRepetitions):
        
            agents = distributedAgents[0:nAgents[jAgents],:]
            nGroups = int(nAgents[jAgents]*nGroupsPartial)
            
            voiPositions = np.ones(nCities)*nVois[iVois]/nCities ### RESETS ALL VOI POSITIONS EVERY DAY (UNIFORMLY)
            
            #Run simulation
            fitnessTemp, maxFitnessTemp, newVoiPositions, nodeUsage = FitnessOfPopulation(voiPositions, nCities, nAgents[jAgents], cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents)
            
            fitnessData[iVois,jAgents] += fitnessTemp/nRepetitions
            maxFitnessData[iVois,jAgents] += maxFitnessTemp/nRepetitions
#Plots
PlotFitness(fitnessData,maxFitnessData,nAgents,nVois,nCities)
plt.show()