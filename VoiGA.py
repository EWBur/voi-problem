import numpy as np

def InitializePopulation(nCities,populationSize):
    population = np.random.rand(populationSize,nCities)
    return population

def DecodePopulation(nVois,population):
    nCities = np.size(population,1)
    decodedPopulation = np.round(population/(np.tile(np.sum(population,1),(nCities,1))).T*nVois)
    
    ### FIX ###
    for i in range(populationSize):
        nVoisInCity = np.sum(decodedPopulation,1)[i]
        while np.sum(decodedPopulation,1)[i] != nVois:
            decodedPopulation[i,np.random.randint(0,nCities)] += np.sign(nVois-nVoisInCity)
    
    return decodedPopulation

def FitnessOfPopulation(population):
    populationFitness = np.linspace(0,100,np.size(population,0))
    return populationFitness

def TournamentSelection(populationFitness,tournamentSize,tournamentProbability):
    populationSize = np.size(populationFitness,0)
    tournamentIndeces = np.random.randint(0,populationSize,tournamentSize)
    #tournamentFitness = np.sort(-populationFitness[tournamentIndeces])
    sortingIndeces = np.argsort(-populationFitness[tournamentIndeces])
    
    probabilities = np.heaviside(-(np.random.rand(tournamentSize) - tournamentProbability),0)
    probabilities[tournamentSize-1] = 1
    possibleIndeces = np.nonzero(probabilities)
    chosenIndex = sortingIndeces[possibleIndeces[0][0]]
    return chosenIndex

def Crossover(chromosome1,chromosome2):
    chromosomeLength = len(chromosome1)
    crossoverPoint = np.random.randint(0,chromosomeLength)
    
    chromosome1New = chromosome1[0:crossoverPoint] + chromosome2[crossoverPoint:-1]
    chromosome2New = chromosome2[0:crossoverPoint] + chromosome1[crossoverPoint:-1]
    return chromosome1New,chromosome2New

def Mutation(chromosome,mutationProbability, creepRate):
    indecesToMutate = np.nonzero(np.heaviside(-(np.random.rand(len(chromosome)) - mutationProbability),0))[0]
    
    for iGene in range(len(indecesToMutate)):
        currentGene = chromosome[iGene]
        mutatedGene = currentGene - creepRate/2 + creepRate*np.random.rand()
        mutatedGene = np.min(np.max(0,mutatedGene),1)
        chromosome[iGene] = mutatedGene
    
    return chromosome

'''
    nCities = len(cities[:,0])
    cityDistancesX = np.tile(cities[:,0],(nCities,1)) - np.tile(cities[:,0],(nCities,1)).T
    cityDistancesY = np.tile(cities[:,1],(nCities,1)) - np.tile(cities[:,1],(nCities,1)).T
    cityDistances = np.sqrt(cityDistancesX**2 + cityDistancesY**2)
    
    cityMap = np.heaviside(-(cityDistances - maxDist),0)
    cityMap = cityMap - np.diag(np.diag(cityMap))

    #nConnections = np.sum(cityMap,0)
    indexOfMinNeighbour = np.argmin(cityDistances + np.diagflat(np.ones(nCities)),1)
    print(indexOfMinNeighbour)
    for i in range(nCities):
        cityMap[indexOfMinNeighbour[i],i] = 1
    '''

nCities = 10
populationSize = 5
nVois = 50
tournamentSize = 5
tournamentProbability = 0.7
mutationProbability = 0.5
creepRate = 0.1

population = InitializePopulation(nCities,populationSize)
decodedPopulation = DecodePopulation(nVois,population)
populationFitness = FitnessOfPopulation(decodedPopulation)
selectedIndecex = TournamentSelection(populationFitness,tournamentSize,tournamentProbability)
chromosome1New,chromosome2New = Crossover([0,1,2,3,4,5],[6,7,8,9])
Mutation(chromosome1New,mutationProbability,creepRate)