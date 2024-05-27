from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import namedtuple
from math import factorial
from utility_functions import (
    arr_bit_to_feature_set,
    feature_set_to_arr,
    string_to_arr,
    arr_bit_to_string)
from SBTN import SBTN
from sklearn.model_selection import cross_val_score
import waterfall_chart
import matplotlib.pyplot as plt
import numpy as np
from model import train_model


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

def get_subsets(A, res, subset, index):
    res.append(subset[:])
    for  i in range(index, len(A)):
        subset.append(A[i])
        get_subsets(A, res, subset, i + 1)
        subset.pop()

def subsets(A):
    subset = []
    res = []
    index = 0
    get_subsets(A, res, subset, index)
    return res

def compute_shapley(selected_features, head_node:SBTN, model, X, y, feature_names = None):
    l = len(selected_features)
    # generate all possible subsets
    print('generating subsets...')
    fs = arr_bit_to_feature_set(selected_features)
    superset = subsets(fs)

    # calculate scores
    total = len(superset)
    print(f"calculating unseen scores... (superset size: {total})")
    cv = []
    to_compute = []
    i = 0
    for subset in superset:
        arr = feature_set_to_arr(subset,l)
        stringified = arr_bit_to_string(arr)
        mean_score = head_node.get_key_score(stringified)
        if mean_score == None:
            if len(subset):
                # calculate scores for newly seen strings
                to_compute.append((stringified, subset))
            else:
                i += 1
                mean_score = 0
                head_node.insert_key(stringified, mean_score)
                cv.append((subset, mean_score))
                print(f'({i}/{total}) \t{subset} score: {mean_score}')
        else:
            i += 1
            cv.append((subset, mean_score))
            print(f'({i}/{total}) \t{subset} score: {mean_score}')

    if len(to_compute):
        with ThreadPoolExecutor(max_workers=len(to_compute)) as executor:
            futures = [executor.submit(train_model, model, X[:,v[1]], y, v[0]) for v in to_compute]

            for future in as_completed(futures):
                score = future.result()
                i += 1
                head_node.insert_key(score[0], score[1])
                mean_score = score[1]
                arr_set = arr_bit_to_feature_set(string_to_arr(score[0]))
                cv.append((arr_set, mean_score))
                print(f'({i}/{total}) \t{arr_set} score: {mean_score}')

    shap = coalitionValues(cv, fs)

    # calculate shapley
    print('calculating shapley values...')
    shapley(shap, feature_names)
