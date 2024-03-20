from __future__ import annotations
import sys

from numpy import array, zeros
sys.path.append('test.py')
sys.path.append('printing.py')
sys.path.append('structs.py')
sys.path.append('display.py')
from test import bukin_n6, easom, goldstein_price, himmelblaus, mccormick, sphere,booth, three_hump_camel
from display import color_map_points
from structs import Point
from general import generate_initial_population, get_value, catch_zero_error

from math import ceil, exp, sqrt
from random import choice, gauss, uniform
from typing import Callable

def calculate_distance(x1, y1, x2, y2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_attraction(
        population:list,
        f_i:tuple,
        beta_0:float = 0.1,
        gamma:float = 0,
        alpha:float = 0.0001,
        elof:array = array([gauss(), gauss()])
        ):
    '''
    NOTE: is an implementation of the main update formula for firefly algorithm.
    The following are descriptions for each parameter
        - beta_0 is the attractiveness coefficient magnitude
        - gamma is the light absorption rate // higher value means higher decay rate with distance
        - alpha controls step-size
        - elof is a vector drawn from a distribution
    '''

    vecSelf = array(f_i[1])
    velocity = zeros(2)
    for f_j in population:
        if f_j[0] > f_i[0]:   # check if the target firefly has higher intensity
            vecTarget = array(f_j[1])
            coeff = beta_0 * exp(-gamma * calculate_distance(*f_i[1],*f_j[1])**2)
            distance = vecTarget - vecSelf
            velocity += coeff * distance

    velocity += vecSelf + alpha*elof # allows the brightest fireflies to move randomly

    return Point([*velocity])

def calculate_intensities(population:list[Point], fitnessFunction:Callable):
    return [ (1/catch_zero_error(fitnessFunction(*p)), p) for p in population ]

# FIREFLY ALGO BASIC IMP
def firefly(fitnessFunction:Callable,
                     generation_count:int,
                     generation_limit:int,
                     generation_x_range:tuple[int,int] = (-1,1),
                     generation_y_range:tuple[int,int] = None,
                     plot_xrange:tuple[int,int] = None,
                     plot_yrange:tuple[int,int] = None,
                     ):

    # Spawn initial point population
    if generation_y_range == None:
        generation_y_range = generation_x_range
    population = generate_initial_population(generation_count,x_range = generation_x_range, y_range = generation_y_range)
    peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]

    # Initial display
    if plot_xrange == None:
        plot_xrange = generation_x_range
    if plot_yrange == None:
        plot_yrange = plot_xrange
    color_map_points(population, fitnessFunction, xrange=plot_xrange, yrange=plot_yrange)
    print(f"loop {-1} best: {get_value(fitnessFunction, peak)}")

    for i in range(generation_limit):
        # attach intensities
        population_intensities = calculate_intensities(population, fitnessFunction)

        # calculate attraction for each firefly
        population = [calculate_attraction(population_intensities, f) for f in population_intensities]
        peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]

        # print(f"loop {i} best: {get_value(fitnessFunction, peak)}")

    # Display
    print(get_value(fitnessFunction,peak))
    color_map_points([peak], fitnessFunction)


if __name__ == "__main__":
    firefly(sphere,
            100,
            100,
            generation_x_range=(-100, 100),
            # generation_y_range=(-3, 4),
            #  plot_xrange=(0,0),
            #  plot_yrange=(0,0),
            )