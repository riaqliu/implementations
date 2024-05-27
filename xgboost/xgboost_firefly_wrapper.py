from copy import deepcopy

import numpy as np
from timeit import default_timer as timer
from math import exp, inf
from scipy import linalg
from random import gauss
from scipy.spatial.distance import hamming

import xgboost as xgb

# personal imports
from utility_functions import (
    load_dataset,
    generate_initial_population,
    string_to_arr,
    discretize,
)
from SBTN import SBTN
from shapley_calc import compute_shapley
from model import compute_scores

np.random.seed(3333)

def main():
    name = "wine"
    print(name)
    X, y, bit_length, feature_names = load_dataset(name)

    # Initialize an empty list to store selected feature indices
    best_bit_string = [ 0 for _ in range(bit_length) ]
    best_score = -1

    # Define the machine learning model (in this case, a Random Forest Classifier)
    model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')
    head_node = SBTN()
    population_count = 20
    population = generate_initial_population(bit_length, population_count)
    streak = 0
    threshold = inf
    beta_0 = 0.06
    gamma = 0.0005
    alpha = 0.3

    last_best_score = -1
    start = timer()
    # Loop
    for loop in range(200):
        current_best_score = -1
        current_best_bit_string = None

        head_node, current_best_bit_string, current_best_score, scored_bit_strings = compute_scores(model, X, y, population, head_node)

        # rank bit strings for optimization
        scored_bit_strings.sort(key=lambda bs: bs[1],reverse=True)

        # Calculate new positions based on attraction
        updated_bit_strings = []
        for b1 in scored_bit_strings:
            b1_vec = np.array(b1[0])
            elof = np.array([gauss() for _ in range(bit_length)])
            velocity = b1_vec + alpha*elof
            for b2 in scored_bit_strings:
                if b2[1] > b1[1]:
                    b2_vec = np.array(b2[0])
                    difference = hamming(b1_vec, b2_vec) * bit_length # b2_vec - b1_vec
                    coeff = beta_0 * exp(-gamma * linalg.norm(difference)**2)
                    velocity += coeff * (b2_vec - b1_vec)
                else:
                    break
            updated_bit_strings.append(discretize(list(velocity)))
        population = deepcopy(updated_bit_strings)

        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
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

if __name__ == "__main__":
    main()