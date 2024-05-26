
from collections import namedtuple
from math import factorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import waterfall_chart

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


def shapley(cv:coalitionValues, feature_names = None):
    sv_sum = 0
    scores = []
    for p in cv.players:
        sv = get_marginal_contribution(cv.cvs, p, cv.players)
        scores.append((p,sv))
        sv_sum += sv

    print_boundary()
    print(f"Feature contributions ({len(cv.players)} players)")
    print_boundary()
    for score in scores:
        print(f"Feature_{score[0]}\t\t{score[1]}")
    print_boundary()
    print(f"SUM: {sv_sum}")

    a,b = tuple(list(l) for l in zip(*scores))
    a = [feature_names[i-1] for i in cv.players] if feature_names is not None else [f'Feature {c}' for c in a]
    print(a)
    print(b)
    print([s[0] for s in scores])
    waterfall_chart.plot(a, b, formatting='{:,.3f}')
    plt.show()

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