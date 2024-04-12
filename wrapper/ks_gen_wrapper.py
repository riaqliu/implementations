from copy import deepcopy
from random import choice, randint, random
from typing import List
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
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


def generate_initial_population(length:int, count:int):
    population = []

    for _ in range(count):
        l = randint(0,length-1)
        new_bit_list = [ 0 for _ in range(length) ]
        new_bit_list[l] = 1
        population.append(new_bit_list)

    return population

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
    population_count = 100
    population = generate_initial_population(bit_length, population_count)
    mutation_chance = 0.5
    parents = None

    # Loop
    for loop in range(1000):
        current_best_score = -1
        current_best_bit_string = None
        ranked_bit_strings = []

        if parents != None:
            # Create new population
            # print('parents', parents)
            population = crossover_and_mutate(parents, population_count, mutation_chance)

            # Cull strings without any feature
            population = [ind for ind in population if 1 in ind]

        # print(population)
        # print([arr_bit_to_feature_set(i) for i in population])

        for bit_string in population:
            # Evaluate the model's performance using cross-validation
            stringified = arr_bit_to_string(bit_string)
            if stringified in evaluated_bit_strings:
                # keep track of already evaluated strings
                mean_score = evaluated_bit_strings.get(stringified)
            else:
                # calculate scores for newly seen strings
                scores = cross_val_score(model, X[:, arr_bit_to_feature_set(bit_string)], y, cv=5, scoring='accuracy')
                mean_score = np.mean(scores)
                evaluated_bit_strings[stringified] = mean_score

            ranked_bit_strings.append((bit_string, mean_score))

            # Keep track of the best-performing feature set
            if mean_score > current_best_score:
                current_best_score = mean_score
                current_best_bit_string = bit_string

        # Apply elitism
        ranked_bit_strings.sort(key=lambda bs: bs[1], reverse=True)
        parents, _ = zip(*ranked_bit_strings[:2])
        parents = list(parents)

        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
                print(f"[{loop}] Current best feature set {arr_bit_to_string(current_best_bit_string)}, Mean Accuracy: {current_best_score:.4f} // saved: {len(evaluated_bit_strings)}")

    print(f"Selected feature indices: {arr_bit_to_feature_set(best_bit_string)} : Mean Accuracy: {best_score:.4f} // saved: {len(evaluated_bit_strings)}")
    # print(evaluated_bit_strings)


if __name__ == "__main__":
    main()
    # Records:
    # @WINE_Database
    # [0, 2, 4, 6, 8, 9, 10, 12] : Mean Accuracy: 0.9889
    # [0, 1, 2, 4, 6, 9, 10, 12] : Mean Accuracy: 0.9889
    # [0, 1, 4, 6, 9, 12] : Mean Accuracy: 0.9944
    # [0, 2, 4, 6, 9] : Mean Accuracy: 0.9889