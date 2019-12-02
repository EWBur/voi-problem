import numpy as np
import matplotlib.pyplot as plt
import TempSim
#%matplotlib qt
import time

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


def FitnessOfPopulation(decodedPopulation, nCities, nAgents, cityMap, cityPositions):
    nIndividuals = np.size(decodedPopulation, 0)
    populationFitness = np.zeros(nIndividuals)
    maxPopulationFitness = np.zeros(nIndividuals)
    
    for vv in range(nIndividuals):
        (populationFitness[vv], maxPopulationFitness[vv]) = TempSim.runSimulation(
            decodedPopulation[vv, :], nCities, nAgents, cityMap, cityPositions)
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


def PlotFitness(noTimeSteps, greatestFitness):
    plt.figure()
    plt.plot(np.linspace(0, noTimeSteps, noTimeSteps+1), greatestFitness, 'k')

    fontSize = 15
    plt.xlabel('Time', fontsize=fontSize)
    plt.ylabel('Greatest fitness', fontsize=fontSize)
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.show()

plt.close("all")

data_set = np.load('MapToUse.npz')
cityMap = data_set['cityMap']
cityPositions = data_set['cityPositions']
nCities = np.size(cityMap,0)

#Model parameters
nAgents = 100
nVois = nCities*2

#GA parameters
noTimeSteps = 300
populationSize = 1
tournamentSize = 2
tournamentProbability = 0.7
mutationProbability = 1/nCities
creepRate = 0.1
elitismNumber = 1
zeroThreshold = 0

population = InitializePopulation(nCities, populationSize)

greatestFitness = np.zeros(noTimeSteps+1)
for iTime in range(noTimeSteps):
    if np.mod(iTime+1, noTimeSteps/10) == 0:
        print('Progress: ' + str((iTime+1)/noTimeSteps*100) + ' %')

    decodedPopulation = DecodePopulation(nVois, population, zeroThreshold)
    populationFitness, maxPopulationFitness = FitnessOfPopulation(
        decodedPopulation, nCities, nAgents, cityMap, cityPositions)

    generationGreatestFitness = np.max(populationFitness)
    if generationGreatestFitness > greatestFitness[iTime]:
        greatestFitness[iTime+1] = generationGreatestFitness
        bestChromosome = population[np.argmax(populationFitness), :]
    else:
        greatestFitness[iTime+1] = greatestFitness[iTime]

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

print(decodedPopulation[0, :])
PlotFitness(noTimeSteps, greatestFitness)
plt.show()
