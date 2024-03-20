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

def calculate_attraction():
    pass


# GEN ALGO BASIC IMP
def firefly(fitnessFunction:Callable,
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
    # plot_points(population, xrange=plot_xrange, yrange=plot_yrange)
    color_map_points(population, fitnessFunction, xrange=plot_xrange, yrange=plot_yrange)

    for i in range(generation_limit):
        print(f"loop {i} best: {get_value(fitnessFunction, peak)}")


    peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]
    print(get_value(fitnessFunction,peak))

    # plot_points([peak], xrange=plot_xrange, yrange=plot_yrange)
    color_map_points([peak], fitnessFunction)


if __name__ == "__main__":

    firefly(mccormick,
                     100,
                     10000,
                     generation_x_range=(-1.5, 4),
                     generation_y_range=(-3, 4),
                    #  plot_xrange=(0,0),
                    #  plot_yrange=(0,0),
                     mutation_chance=0.001,
                     )