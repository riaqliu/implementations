import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd

scores = {
    "Firefly" : [
        2,
        4,
        7,
        5,
        11,
        6,
        7,
        9,
        10
    ],
    "Genetic" : [
        2,
        5,
        3,
        5,
        9,
        4,
        8,
        7,
        6
    ],
}

# Convert scores to DataFrame for plotting
scores_df = pd.DataFrame(scores)
print(scores_df)

# Create a box plot for the cross-validation scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=scores_df)
plt.xlabel("Metaheuristics")
plt.ylabel("Number of Features Selected")
plt.show()