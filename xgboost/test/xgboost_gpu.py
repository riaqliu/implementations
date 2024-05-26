import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Sample data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, size=1000)

# Function to train a model
def train_model(X, y):
    dmatrix = xgb.DMatrix(X, label=y)
    params = {'objective': 'binary:logistic', 'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    model = xgb.train(params, dmatrix, num_boost_round=100)
    return model

# Run multiple classifications in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(train_model, X, y) for _ in range(4)]
    models = [future.result() for future in futures]

print("Trained {} models.".format(len(models)))