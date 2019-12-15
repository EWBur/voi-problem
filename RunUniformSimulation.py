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

def PlotFitnessParameters():
    agents = [50,75,100,125,150,200]
    optimized = [[0.916,0.998,0.998],[0.798,0.947,0.996],[0.760,0.936,0],[0,0,0],[0,0,0],[0,0,0]]
    uniform = [[0.810,0.969,0.999,1.000],[0.733,0.929,0.973,0.995],[0.709,0.899,0.955,0.985],[0.711,0.873,0.939,0.966],[0.669,0.857,0.921,0.961],[0.634,0.806,0.903,0.953]]
    nVoisPerNode = [30,60,90,30,60,90]
    
    markerSize = 20
    lineWidth = 5
    
    plt.figure()
    plt.plot(agents,np.transpose(optimized)[0],'r',marker = 'o',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(optimized)[1],'r',marker = '^',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(optimized)[2],'r',marker = 's',markersize = markerSize,linewidth=lineWidth)
    
    plt.plot(agents,np.transpose(uniform)[0],'k',marker = 'o',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(uniform)[1],'k',marker = '^',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(uniform)[2],'k',marker = 's',markersize = markerSize,linewidth=lineWidth)
        
    fontSize = 40    
        
    plt.xlabel('Number of agents',fontsize=fontSize)
    plt.ylabel('Relative voi usage',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    #plt.title('Actual voi usage relative to maximum voi usage',fontsize=fontSize)
    plt.legend(nVoisPerNode,fontsize=fontSize,frameon=False,title='Number of scooters', title_fontsize=fontSize)

#Import map to use and agents
data_set = np.load('MapToUse4.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
uniformAgents = data_set['uniformAgents']
distributedAgents = data_set['distributedAgents']
nCities = np.size(cityMap,0)

#Model parameters
nAgents = np.asarray([1])
nVois = np.asarray([1])*nCities
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
PlotFitnessParameters()
plt.show()