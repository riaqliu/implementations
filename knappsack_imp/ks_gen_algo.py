from math import inf
import random
from collections import namedtuple
from typing import List

Thing = namedtuple('Thing', ['name', 'value', 'weight'])

def get_items(genome:str, geneList:list[Thing]):
    return [geneList[i][0] for i, v in enumerate(genome) if v == '1']


def generate_initial_population(populationCount:int = 10, genomeLength:int = 10):
    # return [''.join('0' for _ in range(genomeLength)) for _ in range(populationCount)]
    return [''.join(str(random.choice([1,0])) for _ in range(genomeLength)) for _ in range(populationCount)]

def crossover(parents:List[str], generationTarget:int):
    # apply single-point crossover
    alpha, beta = parents[0], parents[1]
    newPopulation = []
    for _ in range(generationTarget):
        r = random.randint(1,len(alpha)-1)
        child = alpha[0:r] + beta[r:]
        newPopulation.append(child)
    return newPopulation

def mutation(generation:List[str], mutationChance:float = 0.5):
    newGeneration = []
    genomeLen = len(generation[0])
    for genome in generation:
        r = random.randint(0,genomeLen-1)
        newGenome = genome[:r] + (genome[r] if random.random() > mutationChance else str(abs(int(genome[r]) - 1)))
        newGenome += genome[r+1:] if r < genomeLen-1 else ''
        newGeneration.append(newGenome)
    return newGeneration

def get_value(genome:str, itemList:list[Thing]):
    if len(genome) != len(itemList):
        raise ValueError("Genome and itemList must be of the same length")
    return sum([itemList[i][1] for i, gene in enumerate(genome) if gene == '1'])

def get_weight(genome:str, itemList:list[Thing]):
    if len(genome) != len(itemList):
        raise ValueError("Genome and itemList must be of the same length")
    return sum([itemList[i][2] for i, gene in enumerate(genome) if gene == '1'])

def fitness_function(genome:str, itemList:list[Thing], weightLimit = inf):
    return get_value(genome, itemList) if get_weight(genome, itemList) <= weightLimit else 0

def ks_genetic(
    geneList:list[Thing],
    populationCount:int = 10,
    generationLimit:int = 10,
    mutationChance:float = 0.5,
    weightLimit:int = inf
):
    # initialize population
    population = generate_initial_population(populationCount, len(geneList))
    # print(population)

    for i in range(generationLimit):
        # select for fitness
        parents = sorted(population, key=lambda genome: fitness_function(genome, geneList, weightLimit), reverse=True)[:2]
        best = parents[0]

        # apply elitism
        print(f"loop {i} best: '{best}' || value: {fitness_function(best, geneList, weightLimit)}")

        # crossover
        population = parents + crossover(parents, populationCount-2)

        # mutate
        population = mutation(population, mutationChance)

    print(f"final best: '{get_items(best, geneList=geneList)}' || value: {get_value(best, geneList)}/ weight: {get_weight(best, geneList)}")



if __name__ == "__main__":
    items = [
        Thing("laptop", 500, 2200),
        Thing("headphones", 150, 160),
        Thing("notepad", 40, 333),
        Thing("coffee mug", 60, 350),
        Thing("water bottle", 30, 192),
    ]

    items2 = [
        Thing("baseball cap", 100, 70),
        Thing("socks", 10, 38),
        Thing("phone", 500, 200),
        Thing("mints", 5, 25),
        Thing("tissues", 15, 80),
    ] + items

    items3 = [
        Thing(f"Thing#{i}", random.randint(1,1000), random.randint(1,1000)) for i in range(5)
    ]

    ks_genetic(
        items2,
        10,
        100,
        weightLimit=3000
    )