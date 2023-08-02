import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, Birch, AgglomerativeClustering, MeanShift
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.mixture import GaussianMixture

# Generate ToyData
blobs_data, _ = make_blobs(n_samples=1000, n_features=2, random_state=42)
classification_data, _ = make_classification(n_samples=1000, n_clusters_per_class=1, n_informative=2, random_state=42)
circles_data, _ = make_circles(n_samples=1000, noise=0.3, factor=0.5, random_state=42)


# Convert the data lists to NumPy arrays
blobs_data = np.array(blobs_data)
classification_data = np.array(classification_data)
circles_data = np.array(circles_data)



# Define clustering algorithms
clustering_algorithms = [
    KMeans(n_clusters=3),
    AffinityPropagation(),
    DBSCAN(),
    GaussianMixture(n_components=3),
    Birch(n_clusters=3),
    AgglomerativeClustering(n_clusters=3),
    MeanShift()
]

# Define dataset names
dataset_names = ["Blobs", "Classification", "Circles"]
datasets = [blobs_data, classification_data, circles_data]

rows = len(clustering_algorithms)
cols = len(datasets)
fig, axes = plt.subplots(rows,cols, figsize=(15, 15))


# Create scatter plots for each algorithm and dataset
for i, algorithm in enumerate(clustering_algorithms):
    for j, dataset in enumerate(datasets):
        algorithm_name = algorithm.__class__.__name__
        dataset_name = dataset_names[j]

        # Fit the algorithm and predict clusters
        clusters = algorithm.fit_predict(dataset)

        # Convert integer labels to colors
        unique_clusters = np.unique(clusters)
        if max(unique_clusters) != 0:
            cluster_colors = [plt.cm.jet(float(i) / max(unique_clusters)) for i in unique_clusters]
        else:
            # Handle the case when max(unique_clusters) is zero
            cluster_colors = ['gray'] * len(unique_clusters)  # Assign gray color to all clusters]
        point_colors = [cluster_colors[cluster] for cluster in clusters]

        # Create scatter plot in the corresponding subplot
        ax = axes[i, j]
        ax.scatter(dataset[:, 0], dataset[:, 1], c=point_colors, s=20)
        ax.set_title(f"{algorithm_name} - {dataset_name}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

# Adjust layout
plt.tight_layout()

# Show the plot

plt.show()