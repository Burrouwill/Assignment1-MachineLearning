from _pydecimal import Decimal, ROUND_HALF_UP

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

# Classifiers
classifiers = {
    'KNN': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'MLPClassifier': MLPClassifier()
}

# Hyperparameter values
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
def calculate_accuracy(classifier, param_name, param_value, x_train, y_train, x_test, y_test):
    all_accuracies = []  # Initialize an empty list to store all accuracies for different parameter values

    # Create a dictionary with the parameter name as the key and the value as the value
    param_dict = {param_name: param_value}

    # Set the hyperparameter using the set_params method with the keyword argument
    clf = classifier.set_params(**param_dict)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    all_accuracies.append(acc)

    return all_accuracies

# Method for rounding to Sf
def round_to_sf(number, sf):
    return Decimal(number).quantize(Decimal('1e-{0}'.format(sf)), rounding=ROUND_HALF_UP)



# Extract features and labels for each dataset
x_steel, labels_steel = datasets["steel-plates-fault"]["data"], datasets["steel-plates-fault"]["target"]
x_ionosphere, labels_ionosphere = datasets["ionosphere"]["data"], datasets["ionosphere"]["target"]
x_banknotes, labels_banknotes = datasets["banknotes"]["data"], datasets["banknotes"]["target"]




# Number of repetitions
num_repetitions = 50
positions_dict = {}

# Data Structs to store stats & figures
plot_list = []
best_mean_errors = {}
best_hyperparams = {}

# Rows
for i, (clf_name, clf) in enumerate(classifiers.items()):
    fig, axs = plt.subplots(1, len(datasets), figsize=(15, 10), sharex='col')
    best_mean_errors[clf_name] = {}
    best_hyperparams[clf_name] = {}
    # Cols
    for j, dataset_name in enumerate(datasets.keys()):
        # Set subplot location
        ax = axs[j]
        # Dict to hold data for plotting
        data = {}
        # Get the param name
        param_name = list(parameter_values[clf_name].keys())[0]
        # Get the list of hyperparams
        param_values = list(parameter_values[clf_name].values())[0]

        for param_value in param_values:
            all_accuracies = []
            all_errors = []
            for _ in range(num_repetitions):
                # Perform train-test split with 50:50 ratio & rand seed
                x_train, x_test, y_train, y_test = train_test_split(datasets[dataset_name]["data"],datasets[dataset_name]["target"], test_size=0.5,random_state=_)

                # Calculate the accuracy & errors for the training data set
                accuracies = calculate_accuracy(clf, param_name, param_value, x_train, y_train, x_test, y_test)
                errors = [1- acc for acc in accuracies]
                all_accuracies.extend(accuracies)
                all_errors.extend(errors)

            # Calc & store accuracies & mean errors
            data[param_value] = all_accuracies
            best_mean_errors[clf_name][dataset_name] = round_to_sf(np.mean(all_errors),4)

        #Find the hyperparam with best mean error
        best_param_value = min(data,key=data.get)
        best_hyperparams[clf_name][dataset_name] = best_param_value

        ax.boxplot(list(data.values()))

        # Create positions for the box plot based on the number of hyperparameter settings
        positions = list(range(1, len(param_values) + 1))
        positions_dict[(clf_name, dataset_name, param_name)] = positions

        # Create a list of strings for the parameter values to use as tick labels
        param_labels = [str(param_value) for param_value in param_values]

        ax.set_xticks(positions)
        ax.set_xticklabels(param_labels)

        ax.set_title(f"{dataset_name} - {clf_name} ({param_name})")
        ax.set_xlabel("Parameter Values")
        ax.set_ylabel("Mean Accuracy")

    # Save plots
    plt.tight_layout()
    plt.savefig(f'{clf_name}_plot.png')
    plot_list.append(plt)










# Create summary table for best mean values of test errors (Table 1)
table1_df = pd.DataFrame(best_mean_errors)
table1_df = table1_df.T
table1_df.index.name = 'Dataset'
table1_df.columns.name = 'Classifier'


# Create summary table for corresponding hyperparameter values (Table 2)
table2_df = pd.DataFrame(best_hyperparams)
table2_df = table2_df.T
table2_df.index.name = 'Dataset'
table2_df.columns.name = 'Classifier'


# Save tables as PNG images
fig_tables = plt.figure(figsize=(12, 6))

# Plot Table 1
ax_tables1 = fig_tables.add_subplot(1, 2, 1)
ax_tables1.axis('off')  # Turn off axis
ax_tables1.table(cellText=table1_df.values,
                 colLabels=table1_df.columns,
                 cellLoc='center',
                 loc='center',
                 colColours=['#f5f5f5'] * len(table1_df.columns),
                 cellColours=[['#f5f5f5'] * len(table1_df.columns)] * len(table1_df),
                 rowLabels=table1_df.index)

# Plot Table 2
ax_tables2 = fig_tables.add_subplot(1, 2, 2)
ax_tables2.axis('off')  # Turn off axis
ax_tables2.table(cellText=table2_df.values,
                 colLabels=table2_df.columns,
                 cellLoc='center',
                 loc='center',
                 colColours=['#f5f5f5'] * len(table2_df.columns),
                 cellColours=[['#f5f5f5'] * len(table2_df.columns)] * len(table2_df),
                 rowLabels=table2_df.index)

# Add titles
plt.suptitle('Summary Tables', fontsize=16)  # This will add a title above both tables
ax_tables1.set_title('Table 1: Best Mean Values of Test Errors')
ax_tables2.set_title('Table 2: Corresponding Hyperparameter Values')
plt.tight_layout()

# Save the tables as a PNG image to a different file
fig_tables.savefig('tables.png', dpi=300)




