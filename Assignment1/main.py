import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
x_steel, labels_steel = datasets["steel-plates-fault"]["data"], datasets["steel-plates-fault"]["target"]
x_ionosphere, labels_ionosphere = datasets["ionosphere"]["data"], datasets["ionosphere"]["target"]
x_banknotes, labels_banknotes = datasets["banknotes"]["data"], datasets["banknotes"]["target"]

# Split data into training and test
# x == data y == labels & train + test
x_train_steel, x_test_steel, y_train_steel, y_test_steel = train_test_split(x_steel, labels_steel, test_size=0.5, random_state=309) #THis should be random?
x_train_ionosphere, x_test_ionosphere, y_train_ionosphere, y_test_ionosphere = train_test_split(x_ionosphere, labels_ionosphere, test_size=0.5, random_state=309) #THis should be random?
x_train_banknotes, x_test_banknotes, y_train_banknotes, y_test_banknotes = train_test_split(x_banknotes, labels_banknotes, test_size=0.5, random_state=309) #THis should be random?


# Setting k for KNN --> Needs to be done k [1,2,3,4,5] for assignment & For all datasets
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train_steel,y_train_steel)
pre_y_knn = neigh.predict(x_test_steel)

# Linear Regression steel: --> Needs to be iterated on for each dataset & for a range of values?
reg = linear_model.LinearRegression()
reg.fit(x_train_steel,y_train_steel)
pre_y_reg = reg.predict(x_test_steel)

#Rideg Regularisation
reg_rid = linear_model.Ridge(alpha=1)
# Train dthe model using training sets
reg_rid.fit(x_train_steel,y_train_steel)
pre_y_rid = reg_rid.predict(x_test_steel)

#NNs for regression
reg_mlp = MLPRegressor(random_state=309,max_iter=1000, alpha=0.0000001)
reg_mlp.fit(x_train_steel, y_train_steel)
pre_y_mlp=reg_mlp.predict(x_test_steel)

# Prediction of Errors / Loss function values
print("Mean Squared Error: %.2f " % mean_squared_error(pre_y_rid,y_test_steel))

print("Coefficient Loss: ", reg_mlp.loss_)


#PCA: Used to project high dimensional data in a low dimensional space
pca = decomposition.PCA(n_components=3)
pca.fit(x_test_steel)
pca_X = pca.transform(x_test_steel)





"""
prin

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
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

datasets_names = ["steel-plates-fault", "ionosphere", "banknotes"]
datasets_accuracies = [steel_accuracies, ionosphere_accuracies, banknotes_accuracies]
k_values = list(range(1, 6))  # k values from 1 to 20

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
"""
