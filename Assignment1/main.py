import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy import sparse

# Get the data:
datasets = {
    "steel-plates-fault": fetch_openml(data_id=1504, as_frame=False, parser='liac-arff'),
    "ionosphere": fetch_openml(data_id=59, as_frame=False, parser='liac-arff'),
    "banknotes": fetch_openml(data_id=1462, as_frame=False, parser='liac-arff')
}

# Define classifiers and datasets
classifiers = {
    'KNN': KNeighborsClassifier(),
    'GNB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'LR': LogisticRegression(),
    'GB': GradientBoostingClassifier(),
    'RF': RandomForestClassifier(),
    'MLP': MLPClassifier()
}

# Define a dictionary to hold the results
results = {}

# Define parameter values for each classifier
parameter_values = {
    'KNN': {'n_neighbors': [3, 5, 7]},
    'GNB': {},  # GaussianNB has no hyperparameters
    'DT': {'max_depth': [None, 5, 10]},
    'LR': {'C': [0.1, 1.0, 10.0]},
    'GB': {'n_estimators': [50, 100, 150]},
    'RF': {'n_estimators': [50, 100, 150]},
    'MLP': {'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)]}
}

# Loop through classifiers and datasets to evaluate accuracy with different parameter values
for clf_name, clf in classifiers.items():
    clf_results = []
    for dataset_name, dataset in datasets.items():
        X, y = dataset.data, dataset.target
        if sparse.issparse(X):
            X = X.toarray()  # Convert sparse matrix to dense array for some classifiers
        clf_scores = []
        if clf_name in parameter_values:
            for param_name, param_values in parameter_values[clf_name].items():
                clf_scores.extend(cross_val_score(clf, X, y, cv=3, param_name=param_name, param_values=param_values))
        else:
            scores = cross_val_score(clf, X, y, cv=3)
            clf_scores.append(scores)
        clf_results.append(clf_scores)
    results[clf_name] = clf_results

# Plot the results as boxplots
fig, axes = plt.subplots(nrows=len(classifiers), ncols=len(datasets), figsize=(15, 20))
for i, clf_name in enumerate(classifiers.keys()):
    for j, dataset_name in enumerate(datasets.keys()):
        ax = axes[i][j]
        param_names = list(parameter_values[clf_name].keys()) if clf_name in parameter_values else ['Default']
        ax.boxplot(results[clf_name][j], labels=param_names)
        ax.set_title(f'{clf_name} - {dataset_name}')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Parameter Values')

plt.tight_layout()
plt.show()
