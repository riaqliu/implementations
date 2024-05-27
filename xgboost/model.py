from concurrent.futures import ThreadPoolExecutor, as_completed
from SBTN import SBTN
import numpy as np
from sklearn.model_selection import cross_val_score
from utility_functions import (
    arr_bit_to_feature_set,
    arr_bit_to_string,
    string_to_arr,
    rounded)

np.random.seed(3333)

def compute_scores(model, X, y, population, head_node:SBTN):
    current_best_score = -1
    current_best_bit_string = None
    to_compute = []
    ranked_bit_strings = []
    for bit_string in population:
        # Evaluate the model's performance using cross-validation
        rounded_bit_string = rounded(bit_string)
        stringified = arr_bit_to_string(rounded_bit_string)
        mean_score = head_node.get_key_score(stringified)
        # raise Exception(mean_score)
        if mean_score == None:
            if 1 in rounded_bit_string:
                # calculate scores for newly seen strings
                to_compute.append((stringified, arr_bit_to_feature_set(rounded_bit_string)))
        # Keep track of the best-performing feature set
        else:
            ranked_bit_strings.append((rounded_bit_string, mean_score))
            if mean_score > current_best_score:
                current_best_score = mean_score
                current_best_bit_string = bit_string
    if len(to_compute):
        with ThreadPoolExecutor(max_workers=min(len(to_compute),2000)) as executor:
            futures = [executor.submit(train_model, model, X[:,v[1]], y, v[0]) for v in to_compute]

            for future in as_completed(futures):
                score = future.result()
                mean_score = head_node.get_key_score(score[0])
                if mean_score == None:
                    head_node.insert_key(score[0], score[1])
                    ranked_bit_strings.append((string_to_arr(score[0]), score[1]))
                    if  score[1] > current_best_score:
                        current_best_score = score[1]
                        current_best_bit_string = score[0]
                else:
                    ranked_bit_strings.append((string_to_arr(score[0]), score[1]))
    return head_node, current_best_bit_string, current_best_score, ranked_bit_strings

def train_model(model, X, y, bitstring):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    return bitstring, np.mean(scores)
