import sys
sys.path.append('/home/trukhinmaksim/src')
from random import sample
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.utils.Cluster import Factory as ClusterFactory

size = (8,6)

DIR = "/home/trukhinmaksim/src/results/plots"
NAME = "BERT_COS"

def plotDistributions(group1, group0, postfix):
    # Sample data (replace with your actual data)cosine_similarity
    plt.figure(figsize=size)
    sns.histplot(group1, color="skyblue", label="Related group (1)", kde=True, stat="density", alpha=0.6, bins=60)
    sns.histplot(group0, color="orange", label="Unrelated group (0)", kde=True, stat="density", alpha=0.6, bins=60)
    #plt.title('RelatePorjects')
    plt.xlabel('Cosine similarity value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(DIR, NAME + f"_distr_{postfix}.png"), dpi = 300)
    #plt.show()

def plotHdb(vectors, min_cluster_size = 2, cluster_selection_epsilon = 0.001, dim = "3d", postfix = "", lims = dict()):
    mod = ClusterFactory.hdbscan(min_cluster_size = min_cluster_size, cluster_selection_epsilon = cluster_selection_epsilon)
    mod.fit_predict(vectors)
    mod.show(dimensions = dim, method = "matplot", savePath = os.path.join(DIR, NAME + f"_hdbscan_{postfix}.png"), size = size, lims = lims)

    return mod.labels

def plotKme(vectors, n_clusters = 3, n_init = 10, dim = "3d", postfix = "", lims = dict()):
    mod = ClusterFactory.kmeans(n_clusters = n_clusters, n_init = n_init)
    mod.fit_predict(vectors)
    mod.show(dimensions = dim, method = "matplot", savePath = os.path.join(DIR, NAME + f"_kmeans_{postfix}.png"), size = size, lims = lims)

    return mod.labels


def plotHeatmap(vectors, labels, path = ""):
    vectors = np.array(vectors)
    labels = np.array(labels)

    print(f"Ground truth labels: {labels}")

    similarity_matrix = cosine_similarity(vectors)
    # For Euclidean distance, lower values mean more similar.
    # distance_matrix = euclidean_distances(item_embeddings)

    print(f"\nShape of similarity matrix: {similarity_matrix.shape}")

# --- 3. Reorder the Matrix based on Ground Truth Labels ---
# This is crucial for visualizing the clusters.
# We'll sort the indices based on the labels.

# Get the sorted indices
    sorted_indices = np.argsort(labels)

# Apply the sorted indices to both rows and columns of the similarity matrix
    reordered_similarity_matrix = similarity_matrix[sorted_indices, :][:, sorted_indices]

# Also reorder the labels for plotting labels (optional)
    reordered_labels = labels[sorted_indices]

    print("\nMatrix reordered based on ground truth labels.")

# --- 4. Plot the Heatmap ---

    plt.figure(figsize=size)
    sns.heatmap(
        reordered_similarity_matrix,
        cmap='viridis', # Choose a colormap. 'viridis', 'magma', 'hot' are good for similarity
        # cmap='coolwarm', # Good for showing positive/negative similarity if scores can be negative
        annot=False,     # Set to True to show similarity values on the heatmap (can be cluttered)
        fmt=".2f",       # Format for annotations if annot=True
        cbar=True,       # Show color bar
        square=True      # Make cells square
    )

#plt.title('Heatmap of Model Similarity Matrix (Reordered by Ground Truth)')
    ##plt.xlabel('Item Index (Reordered)')
    #plt.ylabel('Item Index (Reordered)')

# Optional: Add lines to visually separate the groups
    """
    group_boundaries = np.cumsum([num_items_per_group] * (num_groups -1))
    for boundary in group_boundaries:
        plt.axvline(boundary, color='red', linestyle='--', linewidth=1)
        plt.axhline(boundary, color='red', linestyle='--', linewidth=1)
    """
# Optional: Add group labels if desired (more complex for many items)
# You could manually add text annotations at the center of each block.

    plt.tight_layout()
    plt.savefig(path, dpi = 300)


def build(evaluator, postfix):
    #vecs = generate_clustered_data(n_points_per_cluster=5, n_clusters=2, cluster_std=0.01)[0]
    group1 = evaluator.group1
    group0 = evaluator.group0 

    vecs = evaluator.getPlotVectors()
    #indices = np.random.choice(arr.shape[0], 1000, replace = False)
    #vecs = arr[indices]

    plotDistributions(group1, group0, postfix)

    min_cluster_size = 2
    cluster_selection_epsilon = 0.01
    command = ""
    limCommand = ""
    lims = {}
    while command != "s":
        limCommand = input("lims >>> ")
        if limCommand:
            lims = json.loads(limCommand)
        labels = plotHdb(vecs, min_cluster_size, cluster_selection_epsilon, postfix = postfix, lims = lims)
        command = input(">>> ")
        if command == "s": break
        
        min_cluster_size = int(input("min_cluster_size: "))
        cluster_selection_epsilon = float(input("cluster_selection_epsilon: "))
    plotHeatmap(vecs, labels, path = os.path.join(DIR, NAME + f"_hdbscan_heatmap_{postfix}.png"))

    n_clusters = 3
    n_init = 10
    command = ""
    limCommand = ""
    while command != "s":
        limCommand = input("lims >>> ")
        if limCommand:
            lims = json.loads(limCommand)
        labels = plotKme(vecs, n_clusters, n_init, postfix = postfix, lims = lims)
        command = input(">>> ")
        if command == "s": break
        n_clusters = int(input("n_clusters: "))
        n_init = int(input("n_init: "))

    plotHeatmap(vecs, labels, path = os.path.join(DIR, NAME + f"_kmeans_heatmap_{postfix}.png"))
