import sys
sys.path.append('structs.py')
from structs import Point

from matplotlib import pyplot as plt, cm
import matplotlib
from numpy import array
from typing import Callable


# PRINTING
def plot_points(arr:list[Point], xrange:tuple[int,int]=(-10,10), yrange:tuple[int,int]=(-10,10)):
    data = array(arr)
    x = data[:,0]
    y = data[:,1]

    plt.plot(x, y, color='red', marker='o', linestyle='')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('My Plot')
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.show()

def catch_zero_error(i):
    if i == 0:
        return 1e-99
    return i

def color_map_points(arr:list[Point], fitnessFunction:Callable,  xrange:tuple[int,int]=(-10,10), yrange:tuple[int,int]=(-10,10)):
    # Sample data
    data = array(arr)
    x = data[:,0]
    y = data[:,1]
    z = array([catch_zero_error(fitnessFunction(*dat)) for dat in data])

    # Create colormap
    cmap = matplotlib.colormaps.get_cmap('viridis')  # You can choose any colormap you prefer

    # Normalize the values to range [0,1]
    normalize = plt.Normalize(vmin=min(z), vmax=max(z))

    # Plot
    plt.scatter(x, y, c=z, cmap=cmap, norm=normalize, s=50, alpha=0.8)  # s is marker size, alpha is transparency
    plt.colorbar()  # Add colorbar to show mapping of values to colors
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.show()

def print_points(arr:list[Point], fit_func:Callable):
    line = "====="
    length = 15
    print(''.join([line for _ in range(length)]))
    print("Points")
    for p in arr:
        print(f' {fit_func(*p)} :\t{p}')
    print(''.join([line for _ in range(length)]))