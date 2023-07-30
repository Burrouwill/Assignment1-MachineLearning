import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Get the data:
datasets = {
    "steel-plates-fault": fetch_openml(data_id=1504, as_frame=False, parser='liac-arff'),
    "ionosphere": fetch_openml(data_id=59, as_frame=False, parser='liac-arff'),
    "banknotes": fetch_openml(data_id=1462, as_frame=False, parser='liac-arff')
}

# Define classifiers and datasets
classifiers = {
    'KNN': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'MLPClassifier': MLPClassifier(max_iter=1000)
}

# Define hyperparameter values for each classifier
parameter_values = {
    'KNN': {'n_neighbors': [1, 2, 3, 4, 5]},
    'GaussianNB': {'var_smoothing': [1e-9, 1e-5, 1e-1]},
    'LogisticRegression': {'C': [0.1, 0.5, 1.0, 2.0, 5.0]},
    'DecisionTreeClassifier': {'max_depth': [1, 3, 5, 8, 10]},
    'GradientBoostingClassifier': {'max_depth': [1, 3, 5, 8, 10]},
    'RandomForestClassifier': {'max_depth': [1, 3, 5, 8, 10]},
    'MLPClassifier': {'alpha': [1e-5, 1e-3, 0.1, 10.0]}
}

# Function to calculate accuracy for each parameter value
def calculate_accuracy(classifier, param_name, param_values, x_train, y_train, x_test, y_test):
    accuracies = []
    for value in param_values:
        clf = classifier.set_params(**{param_name: value})
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return accuracies

# Extract features and labels for each dataset
x_steel, labels_steel = datasets["steel-plates-fault"]["data"], datasets["steel-plates-fault"]["target"]
x_ionosphere, labels_ionosphere = datasets["ionosphere"]["data"], datasets["ionosphere"]["target"]
x_banknotes, labels_banknotes = datasets["banknotes"]["data"], datasets["banknotes"]["target"]

# Split data into training and test
x_train_steel, x_test_steel, y_train_steel, y_test_steel = train_test_split(x_steel, labels_steel, test_size=0.5, random_state=309)
x_train_ionosphere, x_test_ionosphere, y_train_ionosphere, y_test_ionosphere = train_test_split(x_ionosphere, labels_ionosphere, test_size=0.5, random_state=309)
x_train_banknotes, x_test_banknotes, y_train_banknotes, y_test_banknotes = train_test_split(x_banknotes, labels_banknotes, test_size=0.5, random_state=309)

# Create a 7-by-3 table of boxplots for classifier accuracy versus parameter values
plt.figure(figsize=(15, 20))
i = 1
box_data = []
for clf_name, clf in classifiers.items():
    data = []
    for dataset_name, (x_train, x_test, y_train, y_test) in zip(["steel-plates-fault", "ionosphere", "banknotes"],
                                                                [(x_train_steel, x_test_steel, y_train_steel, y_test_steel),
                                                                 (x_train_ionosphere, x_test_ionosphere, y_train_ionosphere, y_test_ionosphere),
                                                                 (x_train_banknotes, x_test_banknotes, y_train_banknotes, y_test_banknotes)]):
        param_name = list(parameter_values[clf_name].keys())[0]
        param_values = parameter_values[clf_name][param_name]
        accuracies = calculate_accuracy(clf, param_name, param_values, x_train, y_train, x_test, y_test)
        data.append(accuracies)
    box_data.append(data)
    plt.subplot(7, 3, i)
    plt.boxplot(data, positions=np.arange(1, 4), showfliers=False)
    plt.title(clf_name)
    plt.xticks(np.arange(1, 4), ["steel-plates-fault", "ionosphere", "banknotes"])
    i += 1

plt.tight_layout()
plt.show()

# Create Table 1 and Table 2
table1_data = []
table2_data = []
for clf_name, clf in classifiers.items():
    data1 = []
    data2 = []
    for dataset_name, (x_train, x_test, y_train, y_test) in zip(["steel-plates-fault", "ionosphere", "banknotes"],
                                                                [(x_train_steel, x_test_steel, y_train_steel, y_test_steel),
                                                                 (x_train_ionosphere, x_test_ionosphere, y_train_ionosphere, y_test_ionosphere),
                                                                 (x_train_banknotes, x_test_banknotes, y_train_banknotes, y_test_banknotes)]):
        param_name = list(parameter_values[clf_name].keys())[0]
        param_values = parameter_values[clf_name][param_name]
        accuracies = calculate_accuracy(clf, param_name, param_values, x_train, y_train, x_test, y_test)
        best_accuracy = max(accuracies)
        best_param_value = param_values[np.argmax(accuracies)]
        data1.append(best_accuracy)
        data2.append(best_param_value)
    table1_data.append(data1)
    table2_data.append(data2)

# Display Table 1
table1_df = pd.DataFrame(table1_data, columns=["steel-plates-fault", "ionosphere", "banknotes"], index=classifiers.keys())
print("Table 1 (Best Mean Value of Test Errors):")
print(table1_df)

# Display Table 2
table2_df = pd.DataFrame(table2_data, columns=["steel-plates-fault", "ionosphere", "banknotes"], index=classifiers.keys())
print
