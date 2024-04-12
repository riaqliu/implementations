from copy import deepcopy
from math import exp, inf
from random import gauss, randint
from scipy import linalg
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def arr_bit_to_string(arr):
    return ''.join(['1' if i==1 else '0' for i in arr])

def arr_bit_to_feature_set(arr):
    return [idx for idx, bit in enumerate(arr) if bit == 1]

def rounded(arr):
    return [1 if i >= 0.5 else 0 for i in arr]

def generate_initial_population(length:int, count:int):
    population = []

    for _ in range(count):
        l = randint(0,length-1)
        new_bit_list = [ 0 for _ in range(length) ]
        new_bit_list[l] = 1
        population.append(new_bit_list)

    return population

def calculate_attraction():
    pass

def main():
    dataset = load_wine()
    # Replace this with your dataset and labels
    X = dataset.data
    y = dataset.target
    bit_length = len(dataset.feature_names)

    print('feature names: ',dataset.feature_names)
    print('target names: ', dataset.target_names)

    # Initialize an empty list to store selected feature indices
    best_bit_string = [ 0 for _ in range(bit_length) ]
    best_score = -1

    print(best_bit_string)

    # Define the machine learning model (in this case, a Random Forest Classifier)
    model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    # model = SVC()
    # model = LogisticRegression()
    evaluated_bit_strings = {}
    population_count = 30
    population = generate_initial_population(bit_length, population_count)
    streak = 0
    threshold = inf
    beta_0 = 0.06
    gamma = 0.0005
    alpha = 0.3

    last_best_score = -1
    # Loop
    for loop in range(1000):
        current_best_score = -1
        current_best_bit_string = None
        scored_bit_strings = []

        # Evaluate the model's performance using cross-validation
        for bit_string in population:
            rounded_bit_string = rounded(bit_string)
            stringified = arr_bit_to_string(rounded_bit_string)
            if stringified in evaluated_bit_strings:
                # keep track of already evaluated strings
                mean_score = evaluated_bit_strings.get(stringified)
            else:
                # calculate scores for newly seen strings
                if 1 in rounded_bit_string:
                    scores = cross_val_score(model, X[:, arr_bit_to_feature_set(rounded_bit_string)], y, cv=5, scoring='accuracy')
                    mean_score = np.mean(scores)
                else:
                    # Disregard strings without any features
                    mean_score = 0
                evaluated_bit_strings[stringified] = mean_score
            scored_bit_strings.append((bit_string, mean_score))

            # Keep track of the best-performing feature set
            if mean_score > current_best_score:
                current_best_score = mean_score
                current_best_bit_string = bit_string

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
                    difference = b2_vec - b1_vec
                    coeff = beta_0 * exp(-gamma * linalg.norm(difference)**2)
                    velocity += coeff * difference
                else:
                    break
            updated_bit_strings.append(list(velocity))
        population = deepcopy(updated_bit_strings)


        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
                streak = 0
            if last_best_score == current_best_score:
                    streak += 1

            last_best_score = current_best_score
            print(f"[{loop}] Current best feature set {arr_bit_to_string(rounded(current_best_bit_string))}, Mean Accuracy: {current_best_score:.4f} // saved: {len(evaluated_bit_strings)}")

        if streak > threshold:
            print("Maximum recurring best string value repetition reached!")
            break
    print(f"Selected feature indices: {arr_bit_to_feature_set(rounded(best_bit_string))} : Mean Accuracy: {best_score:.4f} // saved: {len(evaluated_bit_strings)}")


if __name__ == "__main__":
    main()