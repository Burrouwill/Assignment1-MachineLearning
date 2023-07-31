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
    'MLPClassifier': MLPClassifier()
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
    all_accuracies = []  # Initialize an empty list to store all accuracies for different parameter values
    for value in param_values:
        clf = classifier.set_params(**{param_name: value})
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        all_accuracies.append(acc)  # Append the accuracy for this parameter value
    return all_accuracies  # Return a list of accuracies for all parameter values


# Extract features and labels for each dataset
x_steel, labels_steel = datasets["steel-plates-fault"]["data"], datasets["steel-plates-fault"]["target"]
x_ionosphere, labels_ionosphere = datasets["ionosphere"]["data"], datasets["ionosphere"]["target"]
x_banknotes, labels_banknotes = datasets["banknotes"]["data"], datasets["banknotes"]["target"]

x_train_steel, x_test_steel, y_train_steel, y_test_steel = train_test_split(x_steel, labels_steel, test_size=0.5,
                                                                            random_state=309)
x_train_ionosphere, x_test_ionosphere, y_train_ionosphere, y_test_ionosphere = train_test_split(x_ionosphere,
                                                                                                labels_ionosphere,
                                                                                                test_size=0.5,
                                                                                                random_state=309)
x_train_banknotes, x_test_banknotes, y_train_banknotes, y_test_banknotes = train_test_split(x_banknotes,
                                                                                            labels_banknotes,
                                                                                            test_size=0.5,
                                                                                            random_state=309)

x_train_datasets = {
    "steel-plates-fault": x_train_steel,
    "ionosphere": x_train_ionosphere,
    "banknotes": x_train_banknotes
}

x_test_datasets = {
    "steel-plates-fault": x_test_steel,
    "ionosphere": x_test_ionosphere,
    "banknotes": x_test_banknotes
}

y_train_datasets = {
    "steel-plates-fault": y_train_steel,
    "ionosphere": y_train_ionosphere,
    "banknotes": y_train_banknotes
}

y_test_datasets = {
    "steel-plates-fault": y_test_steel,
    "ionosphere": y_test_ionosphere,
    "banknotes": y_test_banknotes
}






# Define the number of repetitions
num_repetitions = 5
positions_dict = {}

# Create the figure and subplots
fig, axs = plt.subplots(len(classifiers), len(datasets), figsize=(15, 10), sharex='col')

#Rows
for i, (clf_name, clf) in enumerate(classifiers.items()):
    #Cols
    for j, dataset_name in enumerate(datasets.keys()):
        ax = axs[i, j]

        for param_name, param_values in parameter_values[clf_name].items():
            all_accuracies = []
            for _ in range(num_repetitions):
                # Perform train-test split with 50:50 ratio and a random state to get reproducible splits
                x_train, x_test, y_train, y_test = train_test_split(datasets[dataset_name]["data"],
                                                                    datasets[dataset_name]["target"],
                                                                    test_size=0.5, random_state=42)
                accuracies = calculate_accuracy(clf, param_name, param_values,
                                                x_train, y_train, x_test, y_test)
                all_accuracies.append(accuracies)

            positions = np.arange(1, len(param_values) + 1)
            positions_dict[(clf_name, dataset_name, param_name)] = positions

            ax.boxplot(all_accuracies, showfliers=False)

            ax.set_title(f"{dataset_name} - {clf_name} ({param_name})", fontsize=8)
            ax.set_xlabel("Parameter Values", fontsize=8)
            ax.set_ylabel("Mean Accuracy", fontsize=8)
            ax.set_xticks(positions)
            ax.set_xticklabels(param_values, fontsize=8)  # Rotate and align x-axis labels

            
plt.tight_layout()  # Adjust spacing and layout
plt.show()




# Create Table 1 and Table 2
table1_data = []
table2_data = []
for clf_name, clf in classifiers.items():
    data1 = []
    data2 = []
    for dataset_name, (x_train, x_test, y_train, y_test) in zip(["steel-plates-fault", "ionosphere", "banknotes"],
                                                                [(x_train_steel, x_test_steel, y_train_steel,
                                                                  y_test_steel),
                                                                 (x_train_ionosphere, x_test_ionosphere,
                                                                  y_train_ionosphere, y_test_ionosphere),
                                                                 (
                                                                 x_train_banknotes, x_test_banknotes, y_train_banknotes,
                                                                 y_test_banknotes)]):
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
table1_df = pd.DataFrame(table1_data, columns=["steel-plates-fault", "ionosphere", "banknotes"],
                         index=classifiers.keys())
print("Table 1 (Best Mean Value of Test Errors):")
print(table1_df)

# Display Table 2
table2_df = pd.DataFrame(table2_data, columns=["steel-plates-fault", "ionosphere", "banknotes"],
                         index=classifiers.keys())
print
