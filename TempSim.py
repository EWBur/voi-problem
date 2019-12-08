import numpy as np
from heapq import heappush, heappop
import math
#np.random.seed(12378911)

## --------------- { GLOBALS } --------------- ##


## --------------- { INIT } --------------- ##
'''
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
        cityMap[minDistPos[0], minDistPos[1]] = 1
    return cityMap
'''

def find_path(agent, city_map, vois):
    [current, start, end] = agent
    h, w = np.shape(city_map)
    place_index_map = []
    for i in range(w):
        place_index_map.append(
            {
                'value': math.inf,
                'visited_cities': [],
                'current_path': []
            }
        )
    pq = []
    heappush(pq, (0, start))
    for i in range(w):
        if not i == start:
            heappush(pq, (math.inf, i))
    while pq:
        value, place_i = heappop(pq)
        current_place = place_index_map[place_i]

        if place_i == end:
            goal = place_index_map[place_i]
            goal['visited_cities'].append(place_i)
            return (goal['value'], goal['visited_cities'])

        neighbourhood_vec = city_map[place_i, :]
        neighbours = np.where(neighbourhood_vec > 0)
        neighbours = neighbours[0]
        for j in neighbours:
            ## THIS IS THE VOI-LOGIC
            if np.any([vois[c] >= 1 for c in current_place['visited_cities'] + [j]]):
                n_value = value + city_map[place_i, j]/2
            else:
                n_value = value + city_map[place_i, j]
            path = (place_i, j)
            place_data = place_index_map[j]
            if n_value < place_data['value']:
                if (place_data['value'], j) in pq:
                    pq.remove((place_data['value'], j))
                places_visited = current_place['visited_cities'].copy()
                current_path = current_place['current_path'].copy()
                current_path.append(path)
                places_visited.append(place_i)
                place_data['value'] = n_value
                place_data['visited_cities'] = places_visited
                heappush(pq, (place_data['value'], j))
    print("No Path found")
    return []

'''
def pathFinding(agent, cityMap, vois, voiUsage):
    (current, start, end) = agent
    G = nx.from_numpy_matrix(cityMap, create_using=nx.DiGraph())
    path = nx.dijkstra_path(G, start, end)
    for c in path[0: -1]:
        if vois[c] > 0:
            voiUsage += 1
            vois[c] -= 1
            vois[end] += 1
            return (path, voiUsage)
    return (path, voiUsage)
'''

def pathFindingDistances(agent, cityMap, vois, voiUsage, maxVoiUsage,reverseDirection):
    (current, start, end) = agent
    if reverseDirection == 1:
        (current, end, start) = agent
        current = end
    
    (cost,path) = find_path([current,start,end], cityMap,vois)
    
    hasVoi = 0
    for nodeIndex in range(len(path[0: -1])):
        currentNode = path[nodeIndex]       #same as c above
        if vois[currentNode] > 0 and hasVoi == 0:
            hasVoi = 1
            vois[currentNode] -= 1
            vois[end] += 1
         
        if hasVoi == 1:
            voiUsage += cityMap[currentNode,path[nodeIndex+1]]
            
        maxVoiUsage += cityMap[currentNode,path[nodeIndex+1]]
    return (path, voiUsage, maxVoiUsage, vois)

def ShuffleAgents(agents,nGroups):
    groupSize = int(np.floor(np.size(agents,0)/nGroups))
    for groupIndex in range(nGroups):
        np.random.shuffle(agents[groupIndex*groupSize:(groupIndex+1)*groupSize])
    return agents

def MutateAgents(agents,nMutations,nNodes):
    nAgents = np.size(agents,0)
    mutationIndeces = np.random.randint(0,nAgents,nMutations)
    
    for iMutation in range(nMutations):
        randomNode = np.random.randint(0,nNodes)
        randomStartEnd = np.random.randint(1,3)    
        agents[mutationIndeces[iMutation],randomStartEnd]  = randomNode   
    return agents

## --------------- { RUNNING } --------------- ##


def runSimulation(voiPositionsInit, nNodes, nAgents, cityMap, cityPositions, agentsInit, nGroups, mutationProbabilityAgents):
    voiUsage = 0
    maxVoiUsage = 0
    
    #Copy the incoming arrays of agents and voiPositions
    voiPositions = np.zeros(30)
    voiPositions[:] = voiPositionsInit[:]
    agents = np.zeros((nAgents,3),int)
    agents[:,:] = agentsInit[:,:]
    
    #Mutate agents start/end node
    nMutations = int(2*np.round(mutationProbabilityAgents*nAgents))
    agents = MutateAgents(agents,nMutations,nNodes)
    
    #Go forward direction (start -> end)
    agents = ShuffleAgents(agents,nGroups)
    for a in agents:
        (path, voiUsage, maxVoiUsage, voiPositions) = pathFindingDistances(a, cityMap, voiPositions, voiUsage, maxVoiUsage,0)  
    
    #Go reverse direction (end -> start)
    agents = ShuffleAgents(agents,nGroups)
    for a in agents:
        (path, voiUsage, maxVoiUsage, voiPositions) = pathFindingDistances(a, cityMap, voiPositions, voiUsage, maxVoiUsage,1)   
    
    return (voiUsage, maxVoiUsage, voiPositions)
