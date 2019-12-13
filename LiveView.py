import numpy as np
import matplotlib.pyplot as plt
import TempSim
import networkx as nx
import matplotlib.animation
#matplotlib.use("Agg")



import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from heapq import heappush, heappop
import math

## --------------- { INIT } --------------- ##

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
    return (path, voiUsage, maxVoiUsage)

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


def PlotGraphAndVois(cityMap,nCities,voiPositions,cityPositions):
    plt.figure()
    G = nx.from_numpy_matrix(cityMap)
    indices = {}
    poss = {}
    for i in range(nCities):
        poss[i] = cityPositions[i]
        indices[i] = i
    labels = {}
    for i in range(nCities):
        labels[i] = voiPositions[i]
    #nx.draw_networkx(G,poss)
    nx.draw(G, poss, labels=indices)


## --------------- { RUNNING } --------------- ##

#Import map to use and agents
data_set = np.load('MapToUseNew.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
uniformAgents = data_set['uniformAgents']
nNodes = np.size(cityMap,0)

#Model parameters
nAgents = 5
nVois = nNodes*2
nTimeSteps = 20
nGroups = nAgents
mutationProbabilityAgents = 0

#Load agents
agents = np.zeros((nAgents,3),int)
agents[0:nAgents,:] = uniformAgents[0:nAgents,:]

#Initial voi distribution (uniform)
voiPositions = np.ones(nNodes)*nVois/nNodes

fitness = np.zeros(nTimeSteps)
maxFitness = np.zeros(nTimeSteps)
voiDistanceFromCenter = np.zeros(nTimeSteps)
voisPerNode = np.zeros((nNodes, nTimeSteps))

G = nx.from_numpy_matrix(cityMap)

indices = {}
poss = {}
for i in range(nNodes):
        poss[i] = cityPositions[i]
        indices[i] = i


Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'), bitrate=1800)

fig,ax = plt.subplots(figsize=(20,10))

ims = []

def update(iTime,*fargs):
    (asd,agents,cityMap,voiPositions) = fargs
    ax.clear()
    #### Go forward direction (start -> end)
    #agents = ShuffleAgents(agents,nGroups)
    for a in agents:
        print(voiPositions)
        (path, voiUsage, maxVoiUsage) = pathFindingDistances(a, cityMap, voiPositions, 0, 0,0)
        node_color = ['green' if voiPositions[n] >= 1 else 'blue' for n in G.nodes]
        for p in path:
            node_color[p] = 'red'
            nx.draw(G, pos=poss,node_color=node_color, ax=ax)
            plt.pause(0.1)
        plt.pause(0.25)

    #### Go reverse direction (end -> start)
    #agents = ShuffleAgents(agents,nGroups)
    #for a in agents:
    #    (path, voiUsage, maxVoiUsage) = pathFindingDistances(a, cityMap, voiPositions, voiUsage, maxVoiUsage,1)


###### LIVE VID -- Remove 'matplotlib.use('Agg')' in imports
def LiveVideo():
    ani = matplotlib.animation.FuncAnimation(fig, update,fargs=(True,agents,cityMap, voiPositions), frames=20, interval=500, repeat=False)
    plt.show()

#### SAVE VID -- ADD 'matplotlib.use('Agg')' in imports
def SaveVideo(fileName):
    with writer.saving(fig,fileName, 100):
        for i in range(5):
            for a in agents:
                (path, voiUsage, maxVoiUsage) = pathFindingDistances(a, cityMap, voiPositions, 0, 0,0)
                node_color = ['green' if voiPositions[n] >= 1 else 'blue' for n in G.nodes]
                for p in path:
                    node_color[p] = 'red'
                    nx.draw(G, pos=poss,node_color=node_color,ax=ax)
                    plt.pause(0.1)
                    writer.grab_frame()


LiveVideo()
#SaveVideo('movieTestLarger.mp4')