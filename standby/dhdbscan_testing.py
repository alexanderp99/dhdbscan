import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../hdbscan")  # Add the submodule directory to the Python path

from hdbscan.hdbscan_ import HDBSCAN
from hdbscan.hdbscan_ import DHDBSCAN
data = np.load('../clusterable_data.npy')

"""
This script was used to look if the cluster that the algorithm produces (sorted by number of points), has the same number of clusterpoints
It turns out: Hdbscan is not shuffle deterministic

"""

def test_determinism(data, n=2, unequal_indices_file='unequal_indices.npy'):
    initial_labels = None
    unequal_indices = []

    # Read unequal indices from file if it exists
    if os.path.exists(unequal_indices_file):
        unequal_indices = np.load(unequal_indices_file).tolist()

    for _ in range(n):
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]

        clusterer = DHDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False).fit(
            shuffled_data)
        reversed_labels = np.zeros_like(clusterer.labels_)
        reversed_labels[shuffled_indices] = clusterer.labels_
        if initial_labels is None:
            initial_labels = reversed_labels
        else:
            unequal_indices.extend(np.where(initial_labels != reversed_labels)[0])
            if not np.array_equal(initial_labels, reversed_labels):
                np.save(unequal_indices_file, unequal_indices)
                return False, unequal_indices
    np.save(unequal_indices_file, unequal_indices)
    return True, unequal_indices

def get_sorted_cluster_sizes(labels):
    cluster_sizes = np.bincount(labels[labels >= 0])
    return sorted(cluster_sizes, reverse=True)

def test_determinism_num_clusters(data, n=2):
    initial_sorted_sizes = None

    for _ in range(n):
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        clusterer = DHDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False, gen_min_span_tree=True).fit(shuffled_data)
        reversed_labels = np.zeros_like(clusterer.labels_)
        reversed_labels[shuffled_indices] = clusterer.labels_
        current_sorted_sizes = get_sorted_cluster_sizes(reversed_labels)
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                              edge_alpha=0.6,
                                              node_size=80,
                                              edge_linewidth=2)
        a = clusterer.minimum_spanning_tree_
        plt.title(f'Single Linkage Tree ')
        plt.show()
        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False
    return True

def test_determinism_num_clusters_without_shuffeling(data, n=2):
    initial_sorted_sizes = None

    for _ in range(n):
        clusterer = DHDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False).fit(data)
        labels = np.copy(clusterer.labels_)
        current_sorted_sizes = get_sorted_cluster_sizes(labels)
        if initial_sorted_sizes is None:
            initial_sorted_sizes = current_sorted_sizes
        else:
            if initial_sorted_sizes != current_sorted_sizes:
                return False
    return True

def test_determinism_without_modifying_unequal_indices(data, n=2):
    initial_labels = None

    for _ in range(n):
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        clusterer = DHDBSCAN(min_cluster_size=15, prediction_data=True, approx_min_span_tree=False).fit(
            shuffled_data)
        reversed_labels = np.zeros_like(clusterer.labels_)
        reversed_labels[shuffled_indices] = clusterer.labels_
        if initial_labels is None:
            initial_labels = reversed_labels
        else:
            if not np.array_equal(initial_labels, reversed_labels):
                return False
    return True


def test_determinism_without_unequal(data, unequal_indices, n=3):
    data_filtered = np.delete(data, unequal_indices, axis=0)
    return test_determinism(data_filtered, n)


"""is_deterministic, unequal_indices = test_determinism(data, 10)
np.random.seed(42)
print("The algorithm is deterministic" if is_deterministic else "The algorithm is not deterministic.")
print(f"Unequal indices: {unequal_indices}")
print(f"Unequal indices length: {len(unequal_indices)}")

is_deterministic = test_determinism_without_unequal(data, unequal_indices)
print("The (modified) algorithm is deterministic" if is_deterministic else "The (modified) algorithm is not deterministic.")
"""
is_deterministic = test_determinism_num_clusters(data,10)
np.random.seed(42)
print("The algorithm is deterministic" if is_deterministic else "The algorithm is not deterministic.")


is_deterministic = test_determinism_num_clusters_without_shuffeling(data,10)
np.random.seed(42)
print("The non shuffled algorithm is deterministic" if is_deterministic else "The non shuffled algorithm is not deterministic.")

