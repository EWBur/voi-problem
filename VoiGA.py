import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt
import time
#import networkx as nx

def InitializePopulation(nCities, populationSize):
    population = np.random.rand(populationSize, nCities)
    return population


def DecodePopulation(nVois, population, zeroThreshold):
    nCities = np.size(population, 1)

    populationCopy = np.zeros((np.size(population, 0), np.size(population, 1)))
    populationCopy[:, :] = population[:, :]
    populationCopy[populationCopy < zeroThreshold] = 0

    decodedPopulation = np.floor(
        population/(np.tile(np.sum(population, 1), (nCities, 1))).T*nVois)

    ### FIX ###
    for i in range(populationSize):
        nVoisInCity = np.sum(decodedPopulation, 1)[i]
        while int(nVoisInCity) != int(nVois):
            decodedPopulation[i, np.random.randint(
                0, nCities)] += np.sign(nVois-nVoisInCity)
            decodedPopulation = np.maximum(decodedPopulation, 0)
            nVoisInCity = np.sum(decodedPopulation, 1)[i]

    return decodedPopulation


def FitnessOfPopulation(decodedPopulation, nCities, nAgents, cityMap, cityPositions, nRepetitions, agents, nGroups,mutationProbabilityAgents,uniformAgents):
    nIndividuals = np.size(decodedPopulation, 0)
    populationFitness = np.zeros(nIndividuals)
    maxPopulationFitness = np.zeros(nIndividuals)
    
    for vv in range(nIndividuals):
        for iRepetition in range(nRepetitions):
            agents[0:nAgents,:] = uniformAgents[0:nAgents,:]
            tempFitness, tempMaxPopulation = TempSim.runSimulation(decodedPopulation[vv, :], nCities, nAgents, cityMap, cityPositions, agents, nGroups, mutationProbabilityAgents)
        
        populationFitness[vv] += tempFitness/nRepetitions
        maxPopulationFitness[vv] += tempMaxPopulation/nRepetitions
    return populationFitness, maxPopulationFitness


def TournamentSelection(populationFitness, tournamentSize, tournamentProbability):
    populationSize = np.size(populationFitness, 0)
    tournamentIndeces = np.random.randint(0, populationSize, tournamentSize)
    #tournamentFitness = np.sort(-populationFitness[tournamentIndeces])
    sortingIndeces = np.argsort(-populationFitness[tournamentIndeces])

    probabilities = np.heaviside(-(np.random.rand(tournamentSize) -
                                   tournamentProbability), 0)
    probabilities[tournamentSize-1] = 1
    possibleIndeces = np.nonzero(probabilities)
    chosenIndex = tournamentIndeces[sortingIndeces[possibleIndeces[0][0]]]
    return chosenIndex


def Crossover(chromosome1, chromosome2):
    chromosomeLength = len(chromosome1)
    crossoverPoint = np.random.randint(0, chromosomeLength)

    chromosome1New = np.concatenate(
        (chromosome1[0:crossoverPoint], chromosome2[crossoverPoint:chromosomeLength]))
    chromosome2New = np.concatenate(
        (chromosome2[0:crossoverPoint], chromosome1[crossoverPoint:chromosomeLength]))
    return chromosome1New, chromosome2New


def Mutation(chromosome, mutationProbability, creepRate):
    indecesToMutate = np.nonzero(
        np.heaviside(-(np.random.rand(len(chromosome)) - mutationProbability), 0))[0]

    for iGene in range(len(indecesToMutate)):
        currentGene = chromosome[iGene]
        mutatedGene = currentGene - creepRate/2 + creepRate*np.random.rand()
        mutatedGene = min(max(0, mutatedGene), 1)
        chromosome[iGene] = mutatedGene

    return chromosome


def PlotFitness(noTimeSteps,greatestFitness):
    plt.figure()
    plt.plot(np.linspace(0, noTimeSteps, noTimeSteps), greatestFitness[1:len(greatestFitness)], 'k')

    fontSize = 20
    plt.xlabel('Time', fontsize=fontSize)
    plt.ylabel('Greatest fitness', fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.title('Greatest fitness of population',fontsize=fontSize)
    plt.show()
    
def PlotGraphAndVois(cityMap,nCities,voiPositions,cityPositions):
    fontSize = 20
    plt.figure()
    G = nx.from_numpy_matrix(cityMap)
    indices = {}
    poss = {}
    for i in range(nCities):
        poss[i] = cityPositions[i]
        indices[i] = i
    labels = {}
    for i in range(nCities):
        labels[i] = int(voiPositions[i])
    nx.draw(G, poss, labels=labels)
    plt.title('Optimal scooter positions',fontsize=fontSize)
    

plt.close("all")

data_set = np.load('MapToUseNew.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
uniformAgents = data_set['uniformAgents']
nCities = np.size(cityMap,0)

#Model parameters
nAgents = 100
nVois = nCities*1

#Simulation parameters
mutationProbabilityAgents = 0
nGroups = nAgents

#GA parameters
nGenerations = 10
nRepetitions = 1
populationSize = 30
tournamentSize = 2
tournamentProbability = 0.7
mutationProbability = 1/nCities
creepRate = 0.1
elitismNumber = 1
zeroThreshold = 0
bestPositionsSaveName = 'BestVoiPositionsTest'

agents = np.zeros((nAgents,3),int)
agents[0:nAgents,:] = uniformAgents[0:nAgents,:]

population = InitializePopulation(nCities, populationSize)

greatestFitness = np.zeros(nGenerations+1)
maxGreatestFitness = np.zeros(nGenerations+1)
for iTime in range(nGenerations):
    if np.mod(iTime+1, nGenerations/10) == 0:
        print('Progress: ' + str((iTime+1)/nGenerations*100) + ' %')

    decodedPopulation = DecodePopulation(nVois, population, zeroThreshold)
    populationFitness, maxPopulationFitness = FitnessOfPopulation(
        decodedPopulation, nCities, nAgents, cityMap, cityPositions,nRepetitions, agents, nGroups, mutationProbabilityAgents,uniformAgents)

    relativePopulationFitness = np.divide(populationFitness,maxPopulationFitness)
    populationFitness = relativePopulationFitness
    generationGreatestFitness = np.max(populationFitness)
    if generationGreatestFitness > greatestFitness[iTime]:
        greatestFitness[iTime+1] = generationGreatestFitness
        maxGreatestFitness[iTime+1] = maxPopulationFitness[np.argmax(populationFitness)]
        bestChromosome = population[np.argmax(populationFitness), :]
    else:
        greatestFitness[iTime+1] = greatestFitness[iTime]
        maxGreatestFitness[iTime+1] = maxGreatestFitness[iTime]

    newPopulation = np.zeros((populationSize, nCities))
    for jChromosomePair in range(int(populationSize/2)):
        chosenIndex1 = TournamentSelection(
            populationFitness, tournamentSize, tournamentProbability)
        chosenIndex2 = TournamentSelection(
            populationFitness, tournamentSize, tournamentProbability)
        chromosome1 = population[chosenIndex1, :]
        chromosome2 = population[chosenIndex2, :]

        chromosome1, chromosome2 = Crossover(chromosome1, chromosome2)

        chromosome1 = Mutation(chromosome1, mutationProbability, creepRate)
        chromosome2 = Mutation(chromosome2, mutationProbability, creepRate)

        newPopulation[2*jChromosomePair, :] = chromosome1
        newPopulation[2*jChromosomePair+1, :] = chromosome2

    newPopulation[0:elitismNumber, :] = bestChromosome
    population = newPopulation

decodedPopulation = DecodePopulation(nVois, population, zeroThreshold)
np.savez(bestPositionsSaveName, bestPositions = np.array(decodedPopulation[0,:]),greatestFitness = np.array(greatestFitness))

print(decodedPopulation[0, :])
PlotFitness(nGenerations, greatestFitness)
#PlotGraphAndVois(cityMap,nCities,decodedPopulation[0,:],cityPositions)
plt.show()
