from copy import deepcopy

import numpy as np
from timeit import default_timer as timer
from math import exp, inf
from scipy import linalg
from random import gauss
from scipy.spatial.distance import hamming
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import cross_val_score

import xgboost as xgb

# personal imports
from utility_functions import (
    load_dataset,
    generate_initial_population,
    string_to_arr,
    discretize,
    arr_bit_to_feature_set,
)
from SBTN import SBTN
from shapley_calc import compute_shapley
from model import compute_scores

np.random.seed(3333)

def calculate_attraction(scored_bit_strings, bit_length):
    beta_0 = 0.06
    gamma = 0.0005
    alpha = 0.3
    updated_bit_strings = []
    if scored_bit_strings:
        with ThreadPoolExecutor(max_workers=min(len(scored_bit_strings), 1000)) as executor:
            futures = [executor.submit(attraction, scored_bit_strings, score, bit_length, beta_0, gamma, alpha) for score in scored_bit_strings]
            for future in as_completed(futures):
                updated_bit_strings.append(future.result())
    return updated_bit_strings

def attraction(scored_bit_strings, score, bit_length, beta_0, gamma, alpha):
    s1_vec = np.array(score[0])
    elof = np.array([gauss() for _ in range(bit_length)])
    velocity = s1_vec + alpha*elof
    for score2 in scored_bit_strings:
        if score2[1] > score[1]:
            s2_vec = np.array(score2[0])
            difference = hamming(s1_vec, s2_vec) * bit_length
            coeff = beta_0 * exp(-gamma * linalg.norm(difference)**2)
            velocity += coeff * (s2_vec - s1_vec)
        else:
            break
    return discretize(list(velocity))

def main():
    name = "QSARbiodegration"
    print(name)
    X, y, bit_length, feature_names = load_dataset(name)

    # Initialize an empty list to store selected feature indices
    best_bit_string = [ 0 for _ in range(bit_length) ]
    best_score = -1
    best_scores = [-1,]

    # Define the machine learning model (in this case, a Random Forest Classifier)
    model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')
    head_node = SBTN()
    population_count = 20
    population = generate_initial_population(bit_length, population_count)
    streak = 0
    threshold = inf

    last_best_score = -1
    start = timer()
    # Loop
    for loop in range(200):
        current_best_score = -1
        current_best_bit_string = None

        head_node, current_best_bit_string, current_best_score, scored_bit_strings, current_best_scores = compute_scores(model, X, y, population, head_node, cv=10)
        # print(scored_bit_strings, current_best_scores, current_best_score)

        # rank bit strings for optimization
        scored_bit_strings.sort(key=lambda bs: bs[1],reverse=True)

        # Calculate new positions based on attraction
        updated_bit_strings = calculate_attraction(scored_bit_strings, bit_length)
        population = deepcopy(updated_bit_strings)

        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
                best_scores = current_best_scores
                streak = 0
                print(f"[{loop}] Current best feature set {current_best_bit_string}, Mean Accuracy: {current_best_score:.4f} // time since last best: {(timer() - start):.4f}s")
                start = timer()
            if last_best_score == current_best_score:
                    streak += 1

            last_best_score = current_best_score

        if streak > threshold:
            print("Maximum recurring best string value repetition reached!")
            break

    print(f"Selected feature indices: {best_bit_string} : Mean Accuracy: {best_score:.4f}")
    compute_shapley(string_to_arr(best_bit_string), head_node, model, X, y, feature_names)
    print(name)
    print(list(best_scores))
    print(np.mean(best_scores))

if __name__ == "__main__":
    main()