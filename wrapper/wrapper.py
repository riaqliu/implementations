from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

wine = load_wine()

# Replace this with your dataset and labels
X = wine.data
y = wine.target

# Initialize an empty list to store selected feature indices
selected_features = []

# Define the machine learning model (in this case, a Random Forest Classifier)
model = RandomForestClassifier()

# Define the number of features you want to select
num_features_to_select = 7

while len(selected_features) < num_features_to_select:
    best_score = -1
    best_feature = None

    for feature_idx in range(X.shape[1]):
        if feature_idx in selected_features:
            continue

        # Try adding the feature to the selected set
        candidate_features = selected_features + [feature_idx]

        # Evaluate the model's performance using cross-validation
        scores = cross_val_score(model, X[:, candidate_features], y, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)

        # Keep track of the best-performing feature
        if mean_score > best_score:
            best_score = mean_score
            best_feature = feature_idx

    if best_feature is not None:
        selected_features.append(best_feature)
        print(f"Selected Feature {len(selected_features)}: {best_feature}, Mean Accuracy: {best_score:.4f}")

print("Selected feature indices:", selected_features)