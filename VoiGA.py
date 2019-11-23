import numpy as np
import matplotlib.pyplot as plt

def InitializePopulation(nCities,populationSize):
    population = np.random.rand(populationSize,nCities)
    return population

def DecodePopulation(nVois,population,zeroThreshold):
    nCities = np.size(population,1)
    
    decodedPopulation = np.round(population/(np.tile(np.sum(population,1),(nCities,1))).T*nVois)
    
    ### FIX ###
    for i in range(populationSize):
        nVoisInCity = np.sum(decodedPopulation,1)[i]
        while int(nVoisInCity) != int(nVois):
            decodedPopulation[i,np.random.randint(0,nCities)] += np.sign(nVois-nVoisInCity)
            decodedPopulation = np.maximum(decodedPopulation,0)
            nVoisInCity = np.sum(decodedPopulation,1)[i]
    
    return decodedPopulation

def FitnessOfPopulation(decodedPopulation):
    populationFitness = np.sum(decodedPopulation[:,0:5],1)
    return populationFitness

def TournamentSelection(populationFitness,tournamentSize,tournamentProbability):
    populationSize = np.size(populationFitness,0)
    tournamentIndeces = np.random.randint(0,populationSize,tournamentSize)
    #tournamentFitness = np.sort(-populationFitness[tournamentIndeces])
    sortingIndeces = np.argsort(-populationFitness[tournamentIndeces])
    
    probabilities = np.heaviside(-(np.random.rand(tournamentSize) - tournamentProbability),0)
    probabilities[tournamentSize-1] = 1
    possibleIndeces = np.nonzero(probabilities)
    chosenIndex = tournamentIndeces[sortingIndeces[possibleIndeces[0][0]]]
    return chosenIndex

def Crossover(chromosome1,chromosome2):
    chromosomeLength = len(chromosome1)
    crossoverPoint = np.random.randint(0,chromosomeLength)

    chromosome1New = np.concatenate((chromosome1[0:crossoverPoint],chromosome2[crossoverPoint:chromosomeLength]))
    chromosome2New = np.concatenate((chromosome2[0:crossoverPoint],chromosome1[crossoverPoint:chromosomeLength]))
    return chromosome1New,chromosome2New

def Mutation(chromosome,mutationProbability, creepRate):
    indecesToMutate = np.nonzero(np.heaviside(-(np.random.rand(len(chromosome)) - mutationProbability),0))[0]
    
    for iGene in range(len(indecesToMutate)):
        currentGene = chromosome[iGene]
        mutatedGene = currentGene - creepRate/2 + creepRate*np.random.rand()
        mutatedGene = min(max(0,mutatedGene),1)
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

nCities = 20
populationSize = 30
nVois = 100
tournamentSize = 2
tournamentProbability = 0.7
mutationProbability = 1/nCities
creepRate = 0.1
noTimeSteps = 1
elitismNumber = 1
zeroThreshold = 0.15

population = InitializePopulation(nCities,populationSize)

greatestFitness = np.zeros(noTimeSteps+1)
for iTime in range(noTimeSteps):
    decodedPopulation = DecodePopulation(nVois,population,zeroThreshold)
    populationFitness = FitnessOfPopulation(decodedPopulation)
    
    generationGreatestFitness = np.max(populationFitness)
    if generationGreatestFitness > greatestFitness[iTime]:
        greatestFitness[iTime+1] = generationGreatestFitness
        bestChromosome = population[np.argmax(populationFitness),:]
    else:
        greatestFitness[iTime+1] = greatestFitness[iTime]
    
    newPopulation = np.zeros((populationSize,nCities))
    for jChromosomePair in range(int(populationSize/2)):
        chosenIndex1 = TournamentSelection(populationFitness,tournamentSize,tournamentProbability)
        chosenIndex2 = TournamentSelection(populationFitness,tournamentSize,tournamentProbability)
        
        chromosome1 = population[chosenIndex1,:]
        chromosome2 = population[chosenIndex2,:]
        
        chromosome1,chromosome2 = Crossover(chromosome1,chromosome2)
        
        chromosome1 = Mutation(chromosome1,mutationProbability,creepRate)
        chromosome2 = Mutation(chromosome2,mutationProbability,creepRate)
        
        newPopulation[2*jChromosomePair,:] = chromosome1
        newPopulation[2*jChromosomePair+1,:] = chromosome2
            
    newPopulation[0:elitismNumber,:] = bestChromosome  
    population = newPopulation
    
plt.plot(np.linspace(0,noTimeSteps,noTimeSteps+1),greatestFitness)
    