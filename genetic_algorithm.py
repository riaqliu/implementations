from __future__ import annotations
import sys
sys.path.append('test.py')
sys.path.append('printing.py')
sys.path.append('structs.py')
sys.path.append('display.py')
from test import bukin_n6, easom, goldstein_price, himmelblaus, mccormick, sphere,booth, three_hump_camel
from display import color_map_points
from structs import Point
from general import generate_initial_population, get_value

from math import ceil
from random import choice, uniform
from typing import Callable


# GEN ALGO BASIC IMP
def crossover(parents:list[Point], generation_count:int):
    # Get all solutions
    solutions = []
    for p in parents:
        solutions.append(p[0])
        solutions.append(p[1])

    # Create new generation with a mixture of parent solutions
    return [Point([choice(solutions), choice(solutions)]) for _ in range(generation_count)]


def mutation(population:list[Point], rand:float = 0.5):
    newPopulation = []
    for p in population:
        newPopulation.append([v + uniform(-rand,rand) for v in p])
    return newPopulation


def genetic(fitnessFunction:Callable,
            generation_count:int,
            generation_limit:int,
            generation_x_range:tuple[int,int] = (-1,1),
            generation_y_range:tuple[int,int] = None,
            mutation_chance:float = 0.01,
            thresholder_count:int = None,
            plot_xrange:tuple[int,int] = None,
            plot_yrange:tuple[int,int] = None,
            ):

    # Spawn initial point population
    if generation_y_range == None:
        generation_y_range = generation_x_range
    population = generate_initial_population(generation_count,x_range = generation_x_range, y_range = generation_y_range)
    peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]

    if plot_xrange == None:
        plot_xrange = generation_x_range
    if plot_yrange == None:
        plot_yrange = plot_xrange
    if thresholder_count == None:
        top = ceil(int(generation_count * 0.2))
    # plot_points(population, xrange=plot_xrange, yrange=plot_yrange)
    color_map_points(population, fitnessFunction, xrange=plot_xrange, yrange=plot_yrange)

    for i in range(generation_limit):
        # select for fitness
        parents = sorted(population, key=lambda ind: fitnessFunction(*ind))[:top]
        print(f"loop {i} best: {get_value(fitnessFunction, peak)}")

        peak = parents[0]

        # crossover population
        population = parents + crossover(parents, generation_count=generation_count-top)

        # mutate population
        population = mutation(population, mutation_chance)
        # plot_points(population,10,10)
    peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]
    print(get_value(fitnessFunction,peak))

    # plot_points([peak], xrange=plot_xrange, yrange=plot_yrange)
    color_map_points([peak], fitnessFunction)


if __name__ == "__main__":

    genetic(mccormick,
            100,
            10000,
            generation_x_range=(-1.5, 4),
            generation_y_range=(-3, 4),
            #  plot_xrange=(0,0),
            #  plot_yrange=(0,0),
            mutation_chance=0.001,
            )