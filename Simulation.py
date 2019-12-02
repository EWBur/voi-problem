import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from heapq import heappush, heappop
import math

## --------------- { GLOBALS } --------------- ##
np.random.seed(12378911)
nNodes = 20
nAgents = 5


## --------------- { INIT } --------------- ##

def initAgents(nAgents):
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


def buildPaths(cities, maxDist):
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

def find_path(agent,city_map, vois):
    [current,start,end] = agent
    has_voi = False
    h,w = np.shape(city_map)
    place_index_map = []
    for i in range (w):
        place_index_map.append(
            {
                'value': math.inf,
                'visited_cities' : [],
                'current_path' : []
            }
        )
    pq = []
    heappush(pq,(0,start))
    for i in range(w):
        if not i == start:
            heappush(pq,(math.inf,i))
    while pq:
        value,place_i = heappop(pq)
        current_place = place_index_map[place_i]

        if place_i == end:
            goal = place_index_map[place_i]
            goal['visited_cities'].append(place_i)
            return (goal['value'],goal['visited_cities'])

        neighbourhood_vec = city_map[place_i,:]
        neighbours = np.where(neighbourhood_vec > 0)
        neighbours = neighbours[0]
        for j in neighbours:
            ## THIS IS THE VOI-LOGIC
            if np.any([ vois[c] >= 1 for c in current_place['visited_cities']+ [j]]):
                n_value = value + city_map[place_i,j]/2
            else:
                n_value = value + city_map[place_i,j]
            path = (place_i,j)
            place_data = place_index_map[j]
            if n_value < place_data['value']:
                if (place_data['value'],j) in pq:
                    pq.remove((place_data['value'],j))
                places_visited = current_place['visited_cities'].copy()
                current_path = current_place['current_path'].copy()
                current_path.append(path)
                places_visited.append(place_i)
                place_data['value'] = n_value
                place_data['visited_cities'] = places_visited
                heappush(pq, (place_data['value'],j))
    print("No Path found")
    return []
    ## We have to do something if there is no path

## --------------- { RUNNING } --------------- ##

cityPositions = np.random.randint(0, high=10, size=(nNodes, 2))
cityMap = buildPaths(cityPositions, 3)
agents = initAgents(nAgents)
vois = np.zeros(20)
vois[19] =1
(cost, path) = find_path([0,0,1],cityMap,vois)
print(cost)
print(path)
G = nx.from_numpy_matrix(cityMap, create_using=nx.DiGraph())
poss = {}
for i in range(nNodes):
    poss[i] = cityPositions[i]
#nx.draw_networkx(G,poss)
nx.draw_networkx(G, poss,nodelist=path)

print(nx.dijkstra_path(G, 0, 1))

#PlotGraph(cityMap, cityPositions)

plt.show()
