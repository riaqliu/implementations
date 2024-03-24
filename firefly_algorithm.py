from __future__ import annotations
import sys

from numpy import array, float64, zeros
import numpy
sys.path.append('test.py')
sys.path.append('printing.py')
sys.path.append('structs.py')
sys.path.append('display.py')
from test import bukin_n6, easom, goldstein_price, himmelblaus, mccormick, sphere,booth, three_hump_camel
from display import color_map_points, print_points
from structs import Point
from general import generate_initial_population, get_value, catch_zero_error

from math import ceil, exp, sqrt
from random import choice, gauss, uniform
from typing import Callable


def calculate_attraction(
        population:list,
        f_i:tuple,
        beta_0:float = 0.1,
        gamma:float = 0.05,
        alpha:float = 0.01,
        elofDist:Callable = gauss
        ):
    '''
    NOTE: is an implementation of the main update formula for firefly algorithm.
    The following are descriptions for each parameter
        - beta_0 is the attractiveness coefficient magnitude // higher value means more attracted to high-intensity fireflies
        - gamma is the light absorption rate // higher value means higher decay rate with distance
        - alpha controls randomization step-size
        - elof is a random vector drawn from a distribution
    '''

    vecSelf = array(f_i[1])
    velocity = zeros(2)
    if elofDist == uniform:
        elof = array([uniform(-1,1), uniform(-1,1)])
    else:
        elof = array([elofDist(), elofDist()])
    for f_j in population:
        if f_j[0] > f_i[0]:   # check if the target firefly has higher intensity
            vecTarget = array(f_j[1])
            difference = vecTarget - vecSelf
            coeff = beta_0 * exp(-gamma * numpy.linalg.norm(difference))
            velocity += coeff * difference

    velocity += vecSelf + alpha*elof # allows the brightest fireflies to move randomly

    return Point([*velocity])


def calculate_intensities(population:list[Point], fitnessFunction:Callable):
    fitnessValues = [(fitnessFunction(*map(float64, p)), p) for p in population]
    lowestValue = max(fitnessValues, key=lambda p: p[0])[0]
    return sorted([ (abs(p[0]-lowestValue), p[1]) for p in fitnessValues ], reverse=True)


# FIREFLY ALGO BASIC IMP
def firefly(fitnessFunction:Callable,
                    generation_count:int,
                    generation_limit:int,
                    generation_x_range:tuple[int,int] = (-1,1),
                    generation_y_range:tuple[int,int] = None,
                    distribution:Callable = gauss,
                    beta_0= 0.5,
                    gamma= 0.05,
                    alpha= 0.01,
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
    # print_points(population, fitnessFunction)

    for i in range(generation_limit):
        # attach intensities
        population_intensities = calculate_intensities(population, fitnessFunction)

        # calculate attraction for each firefly
        population = [calculate_attraction(
            population_intensities,
            f,
            beta_0=beta_0,
            gamma=gamma,
            alpha=alpha,
            elofDist=distribution
            ) for f in population_intensities]
        peak = sorted(population, key=lambda ind: fitnessFunction(*ind))[0]

        print(f"loop {i} best: {get_value(fitnessFunction, peak)}")

    # Display
    print(get_value(fitnessFunction,peak))
    color_map_points([peak], fitnessFunction)


if __name__ == "__main__":
    firefly(easom,
            20,
            10000,
            generation_x_range=(-100,100),
            # generation_y_range=(-3, 4),
            #  plot_xrange=(0,0),
            #  plot_yrange=(0,0),
            distribution=gauss,
            beta_0  = 0.005,
            gamma   = 0.001,
            alpha   = 0.0005,
            )
