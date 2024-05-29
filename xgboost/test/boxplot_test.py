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
        0.96,
        0.9833333333,
        0.9684055271,
        0.8938705159,
        0.8606635071,
        0.7654273831,
        0.843989071,
        0.9342857143,
        0.8271777003,
    ],
    "Genetic" : [
        0.96,
        0.9833333333,
        0.9701754386,
        0.8929336141,
        0.8577628032,
        0.7680451128,
        0.8440860215,
        0.9342857143,
        0.86
    ],
    "Negative" : [
        0.946666666666666,
        0.9722222222,
        0.9490914787,
        0.891053486,
        0.8340521114,
        0.7550239234,
        0.8378494624,
        0.9,
        0.674047619
    ],
}

# Convert scores to DataFrame for plotting
scores_df = pd.DataFrame(scores)
print(scores_df)

# Create a box plot for the cross-validation scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=scores_df)
plt.show()