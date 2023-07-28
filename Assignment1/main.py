import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
}

# Define a dictionary to hold the results
results = {}

# Define hyperparameter values for each classifier
parameter_values = {
    'KNN': {'n_neighbors': [3, 5, 7]}
}

# Extract features and labels for each dataset
X_steel, y_steel = datasets["steel-plates-fault"]["data"], datasets["steel-plates-fault"]["target"]
X_ionosphere, y_ionosphere = datasets["ionosphere"]["data"], datasets["ionosphere"]["target"]
X_banknotes, y_banknotes = datasets["banknotes"]["data"], datasets["banknotes"]["target"]


def knn_hyperparameter_tuning(X, y, k_values=[1, 2, 3, 4, 5], num_splits=50, random_seed=42):
    test_accuracies = []

    for _ in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_seed)

        # Create and train the KNeighborsClassifier with different hyperparameter settings
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            # Evaluate the model on the test set
            accuracy = knn.score(X_test, y_test)
            test_accuracies.append((k, accuracy))

    return test_accuracies


# Perform hyperparameter tuning for each dataset
steel_accuracies = knn_hyperparameter_tuning(X_steel, y_steel)
ionosphere_accuracies = knn_hyperparameter_tuning(X_ionosphere, y_ionosphere)
banknotes_accuracies = knn_hyperparameter_tuning(X_banknotes, y_banknotes)


def plot_boxplot(dataset_accuracies, dataset_name):
    plt.figure(figsize=(10, 6))
    k_values = [entry[0] for entry in dataset_accuracies]
    accuracies = [entry[1] for entry in dataset_accuracies]
    plt.boxplot(accuracies)
    plt.xticks(np.arange(len(k_values)) + 1, k_values)
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Classification Accuracy")
    plt.title(f"KNeighborsClassifier Hyperparameter Tuning for {dataset_name}")
    plt.show()

# Create a table with separate main plots for each dataset
fig, axs = plt.subplots(3, 1, figsize=(8, 20), sharex=True)

datasets_names = ["steel-plates-fault", "ionosphere", "banknotes"]
datasets_accuracies = [steel_accuracies, ionosphere_accuracies, banknotes_accuracies]
k_values = list(range(1, 21))  # k values from 1 to 20

for i, dataset_name in enumerate(datasets_names):
    dataset_accuracies = datasets_accuracies[i]
    accuracies = [entry[1] for entry in dataset_accuracies]
    axs[i].boxplot(accuracies)
    axs[i].set_xticks(np.arange(len(k_values)) + 1)
    axs[i].set_xticklabels(k_values)
    axs[i].set_xlabel("Number of Neighbors (k)")
    axs[i].set_ylabel("Classification Accuracy")
    axs[i].set_title(f"{dataset_name} - KNeighborsClassifier")

plt.tight_layout()
plt.show()
