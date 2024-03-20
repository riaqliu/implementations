from math import cos, exp, pi, sin, sqrt

import numpy


def sphere(*x):
    '''
    Global minimum:
    f(x_1,...,x_n) = f(0,...,0) = 0

    Range: -inf <= x <= inf
    '''
    return numpy.sum([num**2 for num in x])

def booth(x, y):
    '''
    Global minimum:
    f(1,3) = 0

    Range: -10 <= x,y <= 10
    '''
    return (x + 2*y -7)**2 + (2*x + y - 5)**2

def himmelblaus(x, y):
    '''
    Global minimums:
    f(3.0,2.0) = 0,
    f(-2.805118,3.131312) = 0,
    f(-3.779310, -3.283186) = 0,
    f(3.0,2.0) = 0

    Range: -5 <= x,y <= 5
    '''
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def three_hump_camel(x, y):
    '''
    Global minimum:
    f(0,0) = 0

    Range: -5 <= x,y <= 5
    '''
    return 2*x**2 - 1.05*x**4 + (x**6 / 6) + x*y + y**2

def easom(x, y):
    '''
    Global minimum:
    f(PI,PI) = -1

    Range: -100 <= x,y <= 100
    '''
    return -cos(x) * cos(y) * exp( - ((x - pi)**2 + (y - pi)**2) )

def goldstein_price(x, y):
    '''
    Global minimum:
    f(0,-1) = 3
    
    Range: -2 <= x,y <= 2
    '''
    return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

def bukin_n6(x,y):
    '''
    Global minimum:
    f(-10,1) = 0
    
    Range: -15 <= x <= -5, -3 <= y <= 3
    '''
    return 100 * sqrt(abs(y - 0.01*x**2)) + 0.01 * abs(x + 10)

def mccormick(x,y):
    '''
    Global minimum:
    f(-0.54719, -1.54719) = -1.9133

    Range: -1.5 <= x <= 4, -3 <= y <= 4
    '''
    return sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

if __name__ == "__main__":
    print(mccormick(-0.54719, -1.54719))