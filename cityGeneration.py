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
    
plt.close("all")

nNodes = 30
maxDist = 0.22

#np.random.seed(12378911)
#cityPositions = np.random.randint(0, high=10, size=(nNodes, 2))
cityPositions = np.random.rand(nNodes, 2)
cityMap = buildPaths(cityPositions, maxDist, nNodes)

#PlotGraph(cityMap, cityPositions)

np.savez('TestFil', cityMap = np.array(cityMap), cityPositions = np.array(cityPositions))

#data_set = np.load('TestFil.npz')
#cityMap = data_set['cityMap']
#cityPositions = data_set['cityPositions']
PlotGraph(cityMap, cityPositions)