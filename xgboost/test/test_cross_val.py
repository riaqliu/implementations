import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRFClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Accuracy Scores:", np.mean(scores))
