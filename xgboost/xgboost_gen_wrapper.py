from copy import deepcopy
from random import randint, random
from typing import List

from sklearn.model_selection import cross_val_score

import numpy as np
from timeit import default_timer as timer

import xgboost as xgb

# personal imports
from utility_functions import (
    load_dataset,
    generate_initial_population,
    string_to_arr,
    arr_bit_to_feature_set,
)
from SBTN import SBTN
from shapley_calc import compute_shapley
from model import compute_scores

np.random.seed(3333)

def crossover_and_mutate(parents:List, generation_target:int, mutation_chance:float):
    alpha, beta = parents[0], parents[1]
    new_population = parents
    genome_length = len(alpha)
    for _ in range(generation_target):
        # apply single-point crossover
        r = randint(1,genome_length-2)
        child = alpha[:r] + beta[r:]
        new_population.append(child)

    population = []
    for child in new_population:
        # apply single gene mutation
        m = randint(0,genome_length-1)
        new_child = deepcopy(child)
        new_child[m] = child[m] if random() > mutation_chance else abs(child[m]-1)
        population.append(new_child)
    return population

def main():
    name = "minesvsrocks"
    print(name)
    X, y, bit_length, feature_names = load_dataset(name)

    # Initialize an empty list to store selected feature indices
    best_bit_string = [ 0 for _ in range(bit_length) ]
    best_score = -1
    best_scores = [-1,]

    # Define the machine learning model (in this case, a Random Forest Classifier)
    model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')
    head_node = SBTN()
    population_count = 100
    population = generate_initial_population(bit_length, population_count)
    mutation_chance = 0.5
    parents = None

    start = timer()
    # Loop
    for loop in range(1000):
        current_best_score = -1
        current_best_bit_string = None

        if parents != None:
            # Create new population
            population = crossover_and_mutate(parents, population_count, mutation_chance)

            # Cull strings without any feature
            population = [ind for ind in population if 1 in ind]

        head_node, current_best_bit_string, current_best_score, ranked_bit_strings, current_best_scores  = compute_scores(model, X, y, population, head_node)

        # Apply elitism
        ranked_bit_strings.sort(key=lambda bs: bs[1], reverse=True)
        parents, _ = zip(*ranked_bit_strings[:2])
        parents = list(parents)

        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
                best_scores = current_best_scores
                print(f"[{loop}] Current best feature set {current_best_bit_string}, Mean Accuracy: {current_best_score:.4f} // time since last best: {(timer() - start):.4f}s")
                start = timer()

    print(f"Selected feature indices: {best_bit_string} : Mean Accuracy: {best_score:.4f}")

    compute_shapley(string_to_arr(best_bit_string), head_node, model, X, y, feature_names)
    print(name)
    print(list(best_scores))
    print(np.mean(best_scores))

def test():
    # NEGATIVE CONTROL
    name = "minesvsrocks"
    X, y, _, _ = load_dataset(name)
    model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
    score = np.mean(scores)
    print(list(scores))
    print(f"{name} Negative Control,  Mean Accuracy: {score}")

if __name__ == "__main__":
    # test()
    main()