import sys
from typing import Callable
sys.path.append('structs.py')
from structs import Point
from random import uniform

def generate_initial_population(generationCount:int = 10, x_range:tuple[int,int]=(-1,1), y_range:tuple[int,int]=(-1,1)):
    return [ Point([uniform(*x_range),uniform(*y_range),]) for _ in range(generationCount)]

def get_value(type:Callable, point: Point):
    return (type(*point), *point)

def catch_zero_error(i):
    if i == 0:
        return 1e-99
    return i