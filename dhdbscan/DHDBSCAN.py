from sklearn.metrics import pairwise_distances
import numpy as np
class DHDBSCAN:
    def __init__(self):
        self.distance_matrix = None
        self.metric = "euclidean"
        self.p = 2
        self.min_samples = 10
        self.alpha = 1
        return

    def fit(self, X, y=None):
        self.distance_matrix = pairwise_distances(X, metric=self.metric, p=self.p)
        self.mutual_reachability = self.calculate_mutual_reachability(self.distance_matrix, self.min_samples, self.alpha)
        self.minimum_spanning_tree = self.mst_linkage_core(self.mutual_reachability)
        self.single_linkage_tree = self.label(self.minimum_spanning_tree)


    def label(self, minimum_spanning_tree):
        pass

    def calculate_mutual_reachability(self, distance_matrix, min_points=5, alpha=1.0):
        """Compute the weighted adjacency matrix of the mutual reachability
        graph of a distance matrix.

        Parameters
        ----------
        distance_matrix : ndarray, shape (n_samples, n_samples)
            Array of distances between samples.

        min_points : int, optional (default=5)
            The number of points in a neighbourhood for a point to be considered
            a core point.

        Returns
        -------
        mututal_reachability: ndarray, shape (n_samples, n_samples)
            Weighted adjacency matrix of the mutual reachability graph.

        References
        ----------
        .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
           Density-based clustering based on hierarchical density estimates.
           In Pacific-Asia Conference on Knowledge Discovery and Data Mining
           (pp. 160-172). Springer Berlin Heidelberg.
        """
        size = distance_matrix.shape[0]
        min_points = min(size - 1, min_points)
        try:
            core_distances = np.partition(distance_matrix,
                                          min_points,
                                          axis=0)[min_points]
        except AttributeError:
            core_distances = np.sort(distance_matrix,
                                     axis=0)[min_points]

        if alpha != 1.0:
            distance_matrix = distance_matrix / alpha

        stage1 = np.where(core_distances > distance_matrix,
                          core_distances, distance_matrix)
        result = np.where(core_distances > stage1.T,
                          core_distances.T, stage1.T).T
        return result

    def mst_linkage_core(self, distance_matrix):
        n_samples = distance_matrix.shape[0]

        result = np.zeros((n_samples - 1, 3))
        node_labels = np.arange(n_samples, dtype=np.intp)
        current_node = 0
        current_distances = np.full(n_samples, np.inf)
        current_labels = node_labels

        for i in range(1, n_samples):
            label_filter = current_labels != current_node
            current_labels = current_labels[label_filter]
            left = current_distances[label_filter]
            right = distance_matrix[current_node][current_labels]
            current_distances = np.minimum(left, right)

            new_node_index = np.argmin(current_distances)
            new_node = current_labels[new_node_index]
            result[i - 1, 0] = current_node
            result[i - 1, 1] = new_node
            result[i - 1, 2] = current_distances[new_node_index]
            current_node = new_node

        return result