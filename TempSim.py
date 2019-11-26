import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

## --------------- { GLOBALS } --------------- ##


## --------------- { INIT } --------------- ##

def initAgents(nAgents, nNodes):  # AGENTS SHOULD NOT START AND END AT SAME NODE
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
                    cityMap[i, j] = 1
        cityMap[minDistPos[0], minDistPos[1]] = 1
    return cityMap


def pathFinding(agent, cityMap, vois, voiUsage):
    (current, start, end) = agent
    G = nx.from_numpy_matrix(cityMap, create_using=nx.DiGraph())
    path = nx.dijkstra_path(G, start, end)
    for c in path[0: -2]:
        if vois[c] > 0:
            voiUsage += 1
            vois[c] -= 1
            vois[end] += 1
            return (path, voiUsage)
    return (path, voiUsage)


## --------------- { PLOTTING } --------------- ##

def PlotGraph(edges, nodes):
    indecesOfEdges = np.where(edges == 1)
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


## --------------- { RUNNING } --------------- ##


def runSimulation(voiPositions, nNodes, nAgents):
    np.random.seed(12378911)
    cityPositions = np.random.randint(0, high=10, size=(nNodes, 2))
    cityMap = buildPaths(cityPositions, 3, nNodes)
    agents = initAgents(nAgents, nNodes)
    voiUsage = 0
    for a in agents:
        (path, voiUsage) = pathFinding(a, cityMap, voiPositions, voiUsage)
    #print(voiPositions)
    #print(' ------- ')

    return voiUsage


#print(voiUsage)
#PlotGraph(cityMap, cityPositions)