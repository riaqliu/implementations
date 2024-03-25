from collections import namedtuple
from math import exp, inf
import random
from typing import Callable

from numpy import array, float64, zeros, linalg


Thing = namedtuple('Thing', ['name', 'value', 'weight'])

def get_items(genome:str, geneList:list[Thing]):
    return [geneList[i][0] for i, v in enumerate(genome) if v >= 0.5]


def generate_initial_population(populationCount:int = 10, genomeLength:int = 10):
    # return [array([0 for _ in range(genomeLength)]) for _ in range(populationCount)]
    return [array([random.choice([1,0]) for _ in range(genomeLength)]) for _ in range(populationCount)]

def get_value(genome:str, itemList:list[Thing]):
    if len(genome) != len(itemList):
        raise ValueError("Genome and itemList must be of the same length")
    return sum([itemList[i][1] for i, gene in enumerate(genome) if gene >= 0.5])


def get_weight(genome:str, itemList:list[Thing]):
    if len(genome) != len(itemList):
        raise ValueError("Genome and itemList must be of the same length")
    return sum([itemList[i][2] for i, gene in enumerate(genome) if gene >= 0.5])

def fitness_function(genome:str, itemList:list[Thing], weightLimit = inf):
    return get_value(genome, itemList) if get_weight(genome, itemList) <= weightLimit else 0

def calculate_intensities(population:list[str], itemList:list[Thing], weightLimit = inf):
    fitnessValues = [(fitness_function(genome, itemList, weightLimit), genome) for genome in population]
    lowestValue = min(fitnessValues, key=lambda p: p[0])[0]
    return sorted([ (abs(p[0]-lowestValue), p[1]) for p in fitnessValues ], key=lambda v: v[0], reverse=True)

def calculate_attraction(
        population:list,
        f_i:tuple,
        beta_0:float = 0.1,
        gamma:float = 0.05,
        alpha:float = 0.01,
        elofDist:Callable = random.gauss
        ):
    '''
    NOTE: is an implementation of the main update formula for firefly algorithm for the knappsack problem.
    The following are descriptions for each parameter
        - beta_0 is the attractiveness coefficient magnitude // higher value means more attracted to high-intensity fireflies
        - gamma is the light absorption rate // higher value means higher decay rate with distance
        - alpha controls randomization step-size
        - elof is a random vector drawn from a distribution
    '''

    vecSelf = array(f_i[1])
    vecLength = len(vecSelf)
    velocity = zeros(vecLength)
    if elofDist == random.uniform:
        elof = array([random.uniform(0,1) for _ in range(vecLength)])
    elif elofDist == random.gauss:
        elof = array([random.gauss() for _ in range(vecLength)])
    else:
        elof = array([elofDist() for _ in range(vecLength)])

    for f_j in population:
        if f_j[0] > f_i[0]:   # check if the target firefly has higher intensity
            vecTarget = array(f_j[1])
            difference = vecTarget - vecSelf
            coeff = beta_0 * exp(-gamma * linalg.norm(difference))
            velocity += coeff * difference

    velocity += vecSelf + alpha*elof # allows the brightest fireflies to move randomly
    return velocity


def ks_firefly(
    geneList:list[Thing],
    populationCount:int = 10,
    generationLimit:int = 10,
    distribution:Callable = random.gauss,
    beta_0= 0.5,
    gamma= 0.05,
    alpha= 0.01,
    weightLimit:int = inf
    ):

    #Initialize population
    population = generate_initial_population(populationCount=populationCount, genomeLength=len(geneList))
    highest = None

    for i in range(generationLimit):
        # attach intensities
        population_intensities = calculate_intensities(population, geneList, weightLimit)
        # print(population_intensities)

        # # calculate attraction for each firefly
        population = [calculate_attraction(
            population_intensities,
            f,
            beta_0=beta_0,
            gamma=gamma,
            alpha=alpha,
            elofDist=distribution
            ) for f in population_intensities]
        # print(population)
        best = sorted(population, key=lambda ind: fitness_function(ind, geneList, weightLimit))[0]
        print(f"loop {i} best: '{[1 if i>0.5 else 0 for i in best]}' || value: {fitness_function(best, geneList, weightLimit)}")

        if highest is None or fitness_function(highest, geneList, weightLimit) < fitness_function(best, geneList, weightLimit):
            highest = best
    print(f"highest: '{get_items(highest, geneList=geneList)}' || value: {get_value(highest, geneList)}/ weight: {get_weight(highest, geneList)}")


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

    ks_firefly(
        items2,
        10,
        1000,
        beta_0= 0.02,
        gamma= 0.005,
        alpha= 0.1,
        weightLimit=3000
    )