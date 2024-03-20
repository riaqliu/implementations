import sys
sys.path.append('structs.py')
from structs import Point
from random import uniform

def generate_initial_population(generationCount:int = 10, x_range:tuple[int,int]=(-1,1), y_range:tuple[int,int]=(-1,1)):
    return [ Point([uniform(*x_range),uniform(*y_range),]) for _ in range(generationCount)]