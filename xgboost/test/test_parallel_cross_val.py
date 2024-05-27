import xgboost as xgb
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# # Sample data
# X = np.random.rand(1000, 10)
# y = np.random.randint(0, 2, size=1000)

# sklearn
dataset = load_iris()
X = dataset.data
y = dataset.target

np.random.seed(3333)

x_ranges = [[0,],[1,2,3],[0,2],[1,3]]
n = len(x_ranges)

model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')

def train_model2(model, X, y):
    scores = cross_val_score(model, X, y, cv=10)
    return np.mean(scores)

with ThreadPoolExecutor(max_workers=n) as executor:
    futures = [executor.submit(train_model2, model, X[:,v], y) for v in x_ranges]
    scores = [future.result() for future in futures]

print("Trained {} models.".format(len(scores)))
print(scores)