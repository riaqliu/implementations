from collections import namedtuple
from copy import deepcopy
from math import exp, inf
from random import gauss, randint
from scipy import linalg
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from timeit import default_timer as timer
from shapley_test import get_marginal_contribution, shapley
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from scipy.spatial.distance import hamming

np.random.seed(3333)

coalitionValues = namedtuple('CoalitionValues', ['cvs', 'players'])

# ========================================================================================
# Trie implementation vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ========================================================================================

class SBTN():
    '''
        Scored Bit Trie Node
    '''
    features = '01'
    def __init__(self) -> None:
        self.end = False
        self.score = None
        self.children = [None for _ in range(len(self.features))]

    def insert_key(self, key:str, score:int):
        currentNode = self
        for c in key:
            n = SBTN.features.find(c)
            if currentNode.children[n] == None:
                newNode = SBTN()
                currentNode.children[n] = newNode
            currentNode = currentNode.children[n]
        currentNode.end = True
        currentNode.score = score

    def get_key_score(self, key:str):
        currentNode = self
        for c in key:
            n = SBTN.features.find(c)
            if currentNode.children[n] == None:
                return None
            currentNode = currentNode.children[n]
        return currentNode.score if currentNode.end else None

# ========================================================================================
# Trie implementation ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ========================================================================================

def arr_bit_to_string(arr):
    return ''.join(['1' if i==1 else '0' for i in arr])

def arr_bit_to_feature_set(arr):
    return [idx for idx, bit in enumerate(arr) if bit == 1]

def feature_set_to_arr(arr, length):
    return [1 if i in arr else 0 for i in range(length)]

def rounded(arr):
    return [1 if i >= 0.5 else 0 for i in arr]

def discretize(arr):
    return [1/(1+exp(i))for i in arr]

def generate_initial_population(length:int, count:int):
    population = []

    for _ in range(count):
        l = randint(0,length-1)
        new_bit_list = [ 0 for _ in range(length) ]
        new_bit_list[l] = 1
        population.append(new_bit_list)

    return population

# ========================================================================================
# Shapley calculation vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# ========================================================================================

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
    i = 0
    print(f"calculating unseen scores... (superset size: {total})")
    cv = []
    for subset in superset:
        i += 1
        arr = feature_set_to_arr(subset,l)
        stringified = arr_bit_to_string(arr)
        mean_score = head_node.get_key_score(stringified)
        if mean_score == None:
            if len(subset):
                scores = cross_val_score(model, X[:, subset], y, cv=5, scoring='accuracy', n_jobs=-1)
                mean_score = np.mean(scores)
            else:
                mean_score = 0
            head_node.insert_key(stringified, mean_score)
        cv.append((subset, mean_score))
        print(f'({i}/{total}) \t{subset} score: {mean_score}')
    shap = coalitionValues(cv, fs)

    # calculate shapley
    print('calculating shapley values...')
    shapley(shap, feature_names)


# ========================================================================================
# Shapley calculation ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ========================================================================================

def main():
    # # sklearn
    # dataset = load_iris()
    # X = dataset.data
    # y = dataset.target
    # bit_length = len(dataset.feature_names)
    # feature_names = dataset.feature_names

    # sklearn
    dataset = load_wine()
    X = dataset.data
    y = dataset.target
    bit_length = len(dataset.feature_names)
    feature_names = dataset.feature_names

    # # Replace this with your dataset and labels
    # dataset = load_breast_cancer()
    # X = dataset.data
    # y = dataset.target
    # bit_length = len(dataset.feature_names)
    # feature_names = dataset.feature_names

    # # Aids dataset
    # dataset = fetch_ucirepo(id=890) 
    # X = dataset.data.features.values
    # y = dataset.data.targets.values.ravel()
    # feature_names = list(dataset.data.headers)
    # bit_length = len(feature_names) - 2

    # # Ionoshpere
    # dataset = fetch_ucirepo(id=52)
    # X = dataset.data.features.values[:-1]
    # y = dataset.data.targets.values.ravel()[:-1]
    # feature_names = list(dataset.data.headers)
    # bit_length = len(feature_names) - 1 

    # # QSAR Biodegradation
    # feature_names = [f"feature_" + str(n+1) for n in range(42)]
    # df = pd.read_csv("C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\biodeg.csv", delimiter=";", names=feature_names)
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].replace({"RB": 1, "NRB": 0}).values
    # bit_length = len(feature_names) - 1

    # # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
    # path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\diabetes.csv"
    # df = pd.read_csv(path)
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].replace({"RB": 1, "NRB": 0}).values
    # feature_names = df.columns
    # bit_length = len(feature_names) - 1

    # # https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
    # path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\heart.csv"
    # df = pd.read_csv(path)
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].replace({"RB": 1, "NRB": 0}).values
    # feature_names = df.columns
    # bit_length = len(feature_names) - 1

    # # https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
    # dataset = fetch_ucirepo(id=151)
    # X = dataset.data.features.sample(n=40, axis="columns")
    # y = dataset.data.targets.values.ravel()
    # feature_names = X.columns
    # bit_length = len(X.columns)
    # X = X.values

    # # https://archive.ics.uci.edu/dataset/855/tuandromd+(tezpur+university+android+malware+dataset)
    # path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\TUANDROMD.csv"
    # df = pd.read_csv(path)
    # df['Label'] = df['Label'].replace({"malware": 1, "goodware": 0})
    # if df['Label'].isna().sum() > 0:
    #     df = df.dropna(subset=['Label'])
    # X = df.iloc[:, :-1].sample(n=40, axis="columns")
    # feature_names = X.columns
    # bit_length = len(feature_names) - 1
    # y = df.iloc[:, -1].values
    # X = X.values


    # Initialize an empty list to store selected feature indices
    best_bit_string = [ 0 for _ in range(bit_length) ]
    best_score = -1

    # Define the machine learning model (in this case, a Random Forest Classifier)
    # import xgboost as xgb
    # model = xgb.XGBClassifier()
    model = RandomForestClassifier()
    # model = KNeighborsClassifier()
    # model = SVC()
    # model = LogisticRegression()
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
    for loop in range(100):
        current_best_score = -1
        current_best_bit_string = None
        scored_bit_strings = []

        # Evaluate the model's performance using cross-validation
        for bit_string in population:
            rounded_bit_string = rounded(bit_string)
            stringified = arr_bit_to_string(rounded_bit_string)
            mean_score = head_node.get_key_score(stringified)
            if mean_score == None:
                # calculate scores for newly seen strings
                if 1 in rounded_bit_string:
                    scores = cross_val_score(model, X[:, arr_bit_to_feature_set(rounded_bit_string)], y, cv=5, scoring='accuracy', n_jobs=-1)
                    mean_score = np.mean(scores)
                else:
                    mean_score = 0
                head_node.insert_key(stringified, mean_score)
            scored_bit_strings.append((rounded(bit_string), mean_score))

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
                    difference = hamming(b1_vec, b2_vec) * bit_length # b2_vec - b1_vec
                    coeff = beta_0 * exp(-gamma * linalg.norm(difference)**2)
                    velocity += coeff * (b2_vec - b1_vec)
                else:
                    break
            updated_bit_strings.append(list(velocity))
        population = deepcopy(updated_bit_strings)


        if current_best_bit_string is not None:
            if current_best_score > best_score:
                best_bit_string = current_best_bit_string
                best_score = current_best_score
                streak = 0
                print(f"[{loop}] Current best feature set {arr_bit_to_string(rounded(current_best_bit_string))}, Mean Accuracy: {current_best_score:.4f} // time since last best: {(timer() - start):.4f}s")
                start = timer()
            if last_best_score == current_best_score:
                    streak += 1

            last_best_score = current_best_score

        if streak > threshold:
            print("Maximum recurring best string value repetition reached!")
            break

    print(f"Selected feature indices: {arr_bit_to_feature_set(rounded(best_bit_string))} : Mean Accuracy: {best_score:.4f}")
    compute_shapley(rounded(best_bit_string), head_node, model, X, y, feature_names)

if __name__ == "__main__":
    main()