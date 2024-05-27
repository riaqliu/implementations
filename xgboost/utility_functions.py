from random import randint
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from ucimlrepo import fetch_ucirepo
import pandas as pd
from math import exp

def arr_bit_to_string(arr):
    return ''.join(['1' if i==1 else '0' for i in arr])

def arr_bit_to_feature_set(arr):
    return [idx for idx, bit in enumerate(arr) if bit == 1]

def feature_set_to_arr(arr, length):
    return [1 if i in arr else 0 for i in range(length)]

def string_to_arr(arr):
    return [1 if c=='1' else 0 for c in arr]

def rounded(arr):
    return [1 if i >= 0.5 else 0 for i in arr]

def discretize(arr):
    return [(1/(1+exp(-i))) for i in arr]

def generate_initial_population(length:int, count:int):
    population = []

    for _ in range(count):
        l = randint(0,length-1)
        new_bit_list = [ 0 for _ in range(length) ]
        new_bit_list[l] = 1
        population.append(new_bit_list)

    return population

def load_dataset(name: str):
    """
    Available datasets: \n
        - iris\n
        - wine\n
        - breastcancer\n
        - ionosphere\n
        - AIDS175\n
        - QSARbiodegration\n
        - PIMAdiabetes\n
        - heartattack\n
        - minesvsrocks\n
        - malware\n
        - taiwanbankruptcy\n
    Utility function, returns X, y, feature_names, bit_length for each.\n
    """

    if name == "iris":
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
        dataset = load_iris()
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names
        bit_length = len(feature_names)

    elif name == "wine":
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
        dataset = load_wine()
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names
        bit_length = len(feature_names)

    elif name == "breastcancer":
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names
        bit_length = len(feature_names)

    elif name == "ionosphere":
        # https://archive.ics.uci.edu/dataset/52/ionosphere
        dataset = fetch_ucirepo(id=52)
        X = dataset.data.features.values[:-1]
        y_categorical  = dataset.data.targets.values.ravel()[:-1]
        y_series = pd.Series(y_categorical)
        y, _ = pd.factorize(y_series)
        feature_names = list(dataset.data.headers)
        bit_length = len(feature_names) - 1

    elif name == "AIDS175":
        # https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
        dataset = fetch_ucirepo(id=890)
        X = dataset.data.features.values
        y = dataset.data.targets.values.ravel()
        feature_names = list(dataset.data.headers)
        bit_length = len(feature_names) - 2

    elif name == "QSARbiodegration":
        # https://archive.ics.uci.edu/dataset/254/qsar+biodegradation
        feature_names = [f"feature_" + str(n+1) for n in range(42)]
        df = pd.read_csv("C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\biodeg.csv", delimiter=";", names=feature_names)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].replace({"RB": 1, "NRB": 0}).values
        bit_length = len(feature_names) - 1

    elif name == "PIMAdiabetes":
        # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\diabetes.csv"
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values[:-1]
        y = df.iloc[:, -1].values[:-1]
        feature_names = df.columns[:-1]
        bit_length = len(feature_names)

    elif name == "heartattack":
        # https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
        path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\heart.csv"
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values[:-1]
        y = df.iloc[:, -1].values[:-1]
        feature_names = df.columns
        bit_length = len(feature_names) - 1

    elif name == "minesvsrocks":
        # https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks
        dataset = fetch_ucirepo(id=151)
        X = dataset.data.features
        y_categorical = dataset.data.targets.values.ravel()
        y_series = pd.Series(y_categorical)
        y, _ = pd.factorize(y_series)
        feature_names = X.columns
        bit_length = len(feature_names)
        X = X.values

    elif name == "malware":
        # https://archive.ics.uci.edu/dataset/855/tuandromd+(tezpur+university+android+malware+dataset)
        path = "C:\\Users\\jav\\Documents\\School\\4th Year Sem 2\\CMSC 198.2\\implementations\\data\\TUANDROMD.csv"
        df = pd.read_csv(path)
        df['Label'] = df['Label'].replace({"malware": 1, "goodware": 0})
        if df['Label'].isna().sum() > 0:
            df = df.dropna(subset=['Label'])
        X = df.iloc[:, :-1]
        feature_names = X.columns
        bit_length = len(feature_names) - 1
        y = df.iloc[:, -1].values
        X = X.values

    elif name == "taiwanbankruptcy":
        # https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
        dataset = fetch_ucirepo(id=572)
        X = dataset.data.features.sample(axis="columns")
        y = dataset.data.targets.values.ravel()
        feature_names = X.columns
        bit_length = len(X.columns)
        X = X.values

    return X, y, bit_length, feature_names