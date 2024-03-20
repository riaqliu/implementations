import sys
sys.path.append('firefly_algorithm.py')
sys.path.append('genetic_algorithm.py')
from firefly_algorithm import firefly
from genetic_algorithm import genetic


from test import bukin_n6, easom, goldstein_price, himmelblaus, mccormick, sphere, booth, three_hump_camel

genetic(sphere,
        100,    # controls the size of the population per iteration
        10000,  # controls the number of iterations
        generation_x_range=(-100, 100), # this one controls the generation range in the x-axis
        generation_y_range=(-100, 100), # same as above but for the y-axis
        plot_xrange=(-100,100), # this one controls the plot range in the x-axis
        plot_yrange=(-100,100), # ...you get the point
        )