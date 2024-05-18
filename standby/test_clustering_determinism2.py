import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#import hdbscan
#sys.path.append("../hdbscan")  # Add the submodule directory to the Python path
#from dhdbscan.DHDBSCAN import DHDBSCAN
#from hdbscan.hdbscan_ import HDBSCAN

from hdbscan.hdbscan.hdbscan_ import HDBSCAN

data = np.load('../clusterable_data.npy')
ndarrays = []

"""
This script was used to look if the cluster that the algorithm produces (sorted by number of points), has the same number of clusterpoints
It turns out: Hdbscan is not shuffle deterministic

"""

def get_sorted_cluster_sizes(labels):
    cluster_sizes = np.bincount(labels[labels >= 0])
    return sorted(cluster_sizes, reverse=True)


def test_determinism_num_clusters(data, n=2):
    initial_sorted_sizes = None
    for _ in range(n):
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        #clusterer1 = DHDBSCAN().fit(data)
        clusterer = HDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False,
                            gen_min_span_tree=True, algorithm="generic", metric="euclidean").fit(shuffled_data)
        reversed_labels = np.zeros_like(clusterer.labels_)
        reversed_labels[shuffled_indices] = clusterer.labels_
        current_sorted_sizes = get_sorted_cluster_sizes(reversed_labels)


        """clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                              edge_alpha=0.6,
                                              node_size=80,
                                              edge_linewidth=2)"""


        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False
    return True


def test_determinism_num_clusters_without_shuffeling(data, n=2):
    initial_sorted_sizes = None

    for _ in range(n):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False).fit(data)
        labels = np.copy(clusterer.labels_)
        current_sorted_sizes = get_sorted_cluster_sizes(labels)
        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False
    return True


is_deterministic = test_determinism_num_clusters(data, 10)
np.random.seed(42)
print("The algorithm is deterministic" if is_deterministic else "The algorithm is not deterministic.")

is_deterministic = test_determinism_num_clusters_without_shuffeling(data, 10)
np.random.seed(42)
print(
    "The non shuffled algorithm is deterministic" if is_deterministic else "The non shuffled algorithm is not deterministic.")
