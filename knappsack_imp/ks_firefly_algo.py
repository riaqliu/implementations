from collections import namedtuple
from math import inf
import random
from typing import Callable


Thing = namedtuple('Thing', ['name', 'value', 'weight'])

def get_items(genome:str, geneList:list[Thing]):
    return [geneList[i][0] for i, v in enumerate(genome) if v == '1']


def generate_initial_population(populationCount:int = 10, genomeLength:int = 10):
    return [''.join(str(random.choice([1,0])) for _ in range(genomeLength)) for _ in range(populationCount)]


def calculate_intensities():
    pass


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
    print(population)

    for i in range(generationLimit):
        pass


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
        100,
    )