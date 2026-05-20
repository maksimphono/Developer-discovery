import sys
sys.path.append('/home/trukhinmaksim/src')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#import plotly.graph_objects as go
#import plotly.offline as pyo

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from hdbscan import HDBSCAN  # Make sure you have this installed: pip install hdbscan

class Clusterer:
    def __init__(self, model):
        self.clusterModel = model
        self.labels = []
        self.clustersNumber = 0
        self.vectors = []

    def fit_predict(self, vectors):
        self.labels = self.clusterModel.fit_predict(vectors)
        self.vectors = vectors
        self.clustersNumber = len(set(self.labels)) - (1 if -1 in self.labels else 0)  # Count clusters, ignoring noise (-1)
        print(self.labels)
        print(f"Number of clusters found by {self.clusterModel}: {self.clustersNumber}")

    def show(self, groups = None, dimensions = "3d", method = "plotly", savePath = "", perplexity = 30, n_iter = 500, size = (7, 5), lims = dict()): # method = "matplot" | "plotly"
        # projects higher-dimensional points onto 3-dimensional or 2-dimensional space
        vectors = []
        labels = []
        clustersNumber = self.clustersNumber

        if groups != None:
            # only specific clusters should be displayed
            for i, label in enumerate(self.labels):
                if label in groups + [-1]:
                    vectors.append(self.vectors[i])
                    labels.append(label)

            vectors = np.array(vectors)
            labels = np.array(labels)
            clustersNumber = len(groups)
        else:
            # show all clusters
            vectors = self.vectors
            labels = self.labels

        tsneModel = TSNE(n_components = int(dimensions[0]), random_state = 42, perplexity = max(1, min(perplexity, len(vectors) - 1)), n_iter = n_iter)
        reducedVectors = tsneModel.fit_transform(vectors)

        # Use "viridis" color palette, handling noise points (label -1)
        palette = sns.color_palette("viridis", clustersNumber)
        colors = [palette[label] if label >= 0 else (0.5, 0.5, 0.5) for label in labels]  # Noise points are colored gray

        if method == "plotly":
            pass
            """
            data = [
                go.Scatter3d(
                    x = reducedVectors[:, 0],
                    y = reducedVectors[:, 1],
                    z = reducedVectors[:, 2],
                    mode = 'markers',
                    marker = dict(
                        size = 8,
                        color = [f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors],
                        opacity = 1,
                    ),
                    text = [f"Group {label}" if label != -1 else "Noise" for label in labels],  # Hover text
                    hoverinfo = 'text'
                )
            ]

            layout = go.Layout(
                margin = dict(l = 0, r = 0, b = 0, t = 0),  # Remove margins
                title = '3D Visualization of HDBSCAN Clusters (t-SNE)',
                scene = dict(
                    xaxis = dict(title = 'X'),
                    yaxis = dict(title = 'Y'),
                    zaxis = dict(title = 'Z'),
                ),
                legend = dict(
                    title = "Clusters",
                    itemsizing = "constant" # Fixes legend marker size
                )
            )

            fig = go.Figure(data = data, layout = layout)

            if savePath != "":
                # Save the plot to an HTML file (or display it in a notebook)
                pyo.offline.plot(fig, filename = savePath)

            fig.show()
            """
        else:
            fig = plt.figure(figsize = size)
            ax = fig.add_subplot(111, projection = dimensions)

            if len(lims) != 0:
                ax.set_xlim3d(*lims["X"])
                ax.set_ylim3d(*lims["Y"])
                ax.set_zlim3d(*lims["Z"])
            # Create the 3D scatter plot
            scatter = ax.scatter(
                reducedVectors[:, 0],
                reducedVectors[:, 1],
                reducedVectors[:, 2],
                c = colors, 
                s = 60, 
                alpha = 1
            )

            #ax.set_title("3D Visualization of HDBSCAN Clusters (t-SNE)")

            clusterNames = [f"Group {i}" for i in range(clustersNumber)] + ["Noise"]
            legendColors = palette[:clustersNumber] + ['gray'] #Extend color list

            legendElements = []
            for color, name in zip(legendColors, clusterNames):
                legendElements.append(plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = color, markersize = 8, label = name))

            ax.legend(handles = legendElements, title = "Clusters")

            if savePath != "":
                plt.savefig(savePath, dpi = 300)

            #plt.show()


class Factory:
    @classmethod
    def hdbscan(cls, min_cluster_size = 10, cluster_selection_epsilon = 0.3):
        return Clusterer(HDBSCAN(min_cluster_size = min_cluster_size, cluster_selection_epsilon = cluster_selection_epsilon, min_samples = 1))

    @classmethod
    def kmeans(cls, n_clusters = 3, n_init = 10, random_state = 42):
        return Clusterer(KMeans(n_clusters = n_clusters, random_state = random_state, n_init = n_init))

    @classmethod
    def agglomerativeClustering(cls, n_clusters = 3):
        return Clusterer(AgglomerativeClustering(n_clusters = n_clusters))

    @classmethod
    def dbscan(cls, eps = 0.2, min_samples = 7):
        return Clusterer(DBSCAN(eps = eps, min_samples = min_samples))
