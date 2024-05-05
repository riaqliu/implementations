
from collections import namedtuple
from math import factorial
from numba import jit, cuda 

coalitionValues = namedtuple('CoalitionValues', ['cvs', 'players'])

def get_marginal_contribution(cv:list[tuple[set,int]], player, players):
    coalitions, values = zip(*cv)
    contributions = []
    for i1, c in enumerate(coalitions):
        if player in c:
            newCoalition = c.copy()
            newCoalition.remove(player)
            i2 = coalitions.index(newCoalition)
            l = len(newCoalition)
            p = len(players)
            weight = (factorial(l)*factorial(p-l-1))/factorial(p)
            contributions.append((values[i1]-values[i2])*weight)
    return sum(contributions)

def print_boundary():
    print(f"{''.join(['===' for _ in range(20)])}")


def shapley(cv:coalitionValues):
    sv_sum = 0
    print_boundary()
    print("Feature contributions")
    print_boundary()
    for p in cv.players:
        sv = get_marginal_contribution(cv.cvs, p, cv.players)
        sv_sum += sv
        print(f"Feature {p}\t:\t{sv}")
    print_boundary()
    print(f"SUM: {sv_sum}")

if __name__ == "__main__":
    p1 = {1,2}
    p2 = {1,2,3}
    cv1 = [
        ({1,2}, 10000),
        ({1}, 7500),
        ({2}, 5000),
        (set(), 0)
    ]
    cv2 = [
        ({1,2,3}, 10000),
        (set(), 0),
        ({1,2}, 7500),
        ({1,3}, 7500),
        ({2,3}, 5000),
        ({1}, 5000),
        ({2}, 5000),
        ({3}, 0),
    ]

    cv1 = coalitionValues(cv1, p1)
    cv2 = coalitionValues(cv2, p2)

    shapley(cv2)

    # a = [0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.16666666666666666]
    # print(sum(a))