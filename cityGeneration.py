import numpy as np
import matplotlib.pyplot as plt

def buildPaths(cities, maxDist, nNodes):
    cityMap = np.zeros((nNodes, nNodes))
    for i in range(len(cities)):
        currentCityPos = cities[i, :]
        minDist = 100000
        minDistPos = [None]*2
        for j in range(len(cities)):
            nextCityPos = cities[j, :]
            if i != j:
                distance = np.sqrt((currentCityPos[0] - nextCityPos[0])**2 +
                                   (currentCityPos[1] - nextCityPos[1])**2)
                if distance < minDist:
                    minDist = distance
                    minDistPos[0] = i
                    minDistPos[1] = j
                if distance <= maxDist:
                    #cityMap[i, j] = 1
                    cityMap[i, j] = distance
        cityMap[minDistPos[0], minDistPos[1]] = minDist
    return cityMap

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
    
def initAgents(nAgents,nNodes,cityPositions):
    agents = np.zeros((nAgents, 3), dtype=np.int8)
    for i in range(nAgents):
        cityIndexes = [x for x in range(nNodes)]
        startCity = np.random.choice(cityIndexes, 1)
        currentCity = startCity
        cityIndexes.remove(startCity)
        networkCenter = FindGraphCenter(cityPositions)
        possibleEndNodes = np.delete(cityPositions,startCity,axis=0)
        centerDistances = FindDistancesToCenter(possibleEndNodes,networkCenter)
        endNodeProbability = GetEndNodeProbability(centerDistances)
        endCity = np.random.choice(cityIndexes, 1, p = endNodeProbability)
        agents[i, 0] = currentCity
        agents[i, 1] = startCity
        agents[i, 2] = endCity
    return agents

def FindGraphCenter(nodePositions):
    networkCenter = np.sum(nodePositions,0)/np.size(nodePositions,0)
    return networkCenter

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
    
plt.close("all")

nNodes = 30
maxDist = 0.22
nAgents = 5000

np.random.seed(12378911)
cityPositions = np.random.randint(0, high=10, size=(nNodes, 2))
cityPositions = np.random.rand(nNodes, 2)
cityMap = buildPaths(cityPositions, maxDist, nNodes)
distributedAgents = initAgents(nAgents, nNodes,cityPositions)
print(distributedAgents)


np.savez('TestAgentDistribution', cityMap = np.array(cityMap), cityPositions = np.array(cityPositions), distributedAgents = distributedAgents)

#data_set = np.load('Test2.npz')
#cityMap = data_set['cityMap']
#cityPositions = data_set['cityPositions']
#PlotGraph(cityMap, cityPositions)