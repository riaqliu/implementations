import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Collect cross-validation scores
scores = {model_name: cross_val_score(model, X, y, cv=10)
          for model_name, model in models.items()}

# Convert scores to DataFrame for plotting
scores_df = pd.DataFrame(scores)
print(scores_df)

# Create a box plot for the cross-validation scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=scores_df)
plt.title('Box plot of cross-validation scores')
plt.show()