import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt

def FindDistancesToCenter(cityPositions,networkCenter):
    xValues = cityPositions[:,0] - networkCenter[0]
    yValues = cityPositions[:,1] - networkCenter[1]
    centerDistances = np.sqrt(
        np.power(xValues,2) + np.power(yValues,2))
    return centerDistances

def GetEndNodeProbability(centerDistances):
    inverseDistance = 1/centerDistances
    totalInverseDistance = sum(inverseDistance)
    endNodeProbability = inverseDistance/totalInverseDistance
    return endNodeProbability

def FindGraphCenter(nodePositions):
    networkCenter = np.sum(nodePositions,0)/np.size(nodePositions,0)
    return networkCenter

def VoiDistanceFromCenter(nodePositions,voiPositions,networkCenter):
    voiDistanceFromCenter = np.matmul(voiPositions,np.sqrt(np.sum((nodePositions-networkCenter)**2,1)))/np.sum(voiPositions)
    return voiDistanceFromCenter

def FitnessOfPopulation(voiPositions, nCities, nAgents, cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents,endNodeProbabilities):
    fitness, maxFitness, newVoiPositions, nodeUsage = TempSim.runSimulation(voiPositions, nCities, nAgents, cityMap, cityPositions, agents, nGroups, mutationProbabilityAgents,endNodeProbabilities)
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
    agents = [50,75,100,125,150,200,250,300]
    nVoisPerNode = [30,60,90,30,60,90]
    
    optimized = [[0.916,0.998,0.998],[0.798,0.947,0.996],[0.760,0.936,0.987],[0.740,0.925,0.977],[0.716,0.888,0.975],[0.663,0.848,0.953],[0.657,0.841,0.935],[0.648,0.818,0.907]]
    uniform = [[0.810,0.969,0.999,1.000],[0.733,0.929,0.973,0.995],[0.709,0.899,0.955,0.985],[0.711,0.873,0.939,0.966],[0.669,0.857,0.921,0.961],[0.634,0.806,0.903,0.953],[0.642,0.809,0.898,0.947],[0.633,0.787,0.874,0.934]]
    
    optimized0101 = [[0.914,0.988,0.994],[0.792,0.972,0.993],[0.745,0.939,0.982],[0.720,0.917,0.970],[0.704,0.885,0.968],[0.671,0.850,0.947],[0.658,0.837,0.932],[0.646,0.813,0.901]]
    uniform0101 = [[0.845,0.975,0.997],[0.757,0.939,0.979],[0.712,0.906,0.962],[0.705,0.882,0.946],[0.682,0.857,0.932],[0.647,0.823,0.914],[0.648,0.816,0.905],[0.638,0.792,0.884]]
    
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
    plt.ylabel('Relative scooter usage',fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    #plt.title('Actual voi usage relative to maximum voi usage',fontsize=fontSize)
    plt.legend(nVoisPerNode,fontsize=fontSize,frameon=False,title='Number of scooters', title_fontsize=fontSize)
    
    plt.figure()
    plt.plot(agents,np.transpose(optimized0101)[0],'r',marker = 'o',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(optimized0101)[1],'r',marker = '^',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(optimized0101)[2],'r',marker = 's',markersize = markerSize,linewidth=lineWidth)
    
    plt.plot(agents,np.transpose(uniform0101)[0],'k',marker = 'o',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(uniform0101)[1],'k',marker = '^',markersize = markerSize,linewidth=lineWidth)
    plt.plot(agents,np.transpose(uniform0101)[2],'k',marker = 's',markersize = markerSize,linewidth=lineWidth)
        
    fontSize = 40    
        
    plt.xlabel('Number of agents',fontsize=fontSize)
    plt.ylabel('Relative scooter usage',fontsize=fontSize)
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

#Compute the graphs center
networkCenter = FindGraphCenter(cityPositions)
centerDistances = FindDistancesToCenter(cityPositions,networkCenter)
endNodeProbabilities = GetEndNodeProbability(centerDistances)

fitnessData = np.zeros((len(nVois),len(nAgents)))
maxFitnessData = np.zeros((len(nVois),len(nAgents)))

for iVois in range(len(nVois)):
    for jAgents in range(len(nAgents)):
        for kRepetitions in range(nRepetitions):
        
            agents = distributedAgents[0:nAgents[jAgents],:]
            nGroups = int(nAgents[jAgents]*nGroupsPartial)
            
            voiPositions = np.ones(nCities)*nVois[iVois]/nCities ### RESETS ALL VOI POSITIONS EVERY DAY (UNIFORMLY)
            
            #Run simulation
            fitnessTemp, maxFitnessTemp, newVoiPositions, nodeUsage = FitnessOfPopulation(voiPositions, nCities, nAgents[jAgents], cityMap, cityPositions,agents, nGroups, mutationProbabilityAgents,endNodeProbabilities)
            
            fitnessData[iVois,jAgents] += fitnessTemp/nRepetitions
            maxFitnessData[iVois,jAgents] += maxFitnessTemp/nRepetitions
#Plots
PlotFitness(fitnessData,maxFitnessData,nAgents,nVois,nCities)
PlotFitnessParameters()
plt.show()