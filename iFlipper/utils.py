from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import falconn

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_auc_score
from aif360.metrics import ClassificationMetric


def measure_error(label, edge, w_edge):
    """         
        Computes the total error in the data.
        Args:
            label: Labels of the data
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
        Return:
            total_error: The total error
    """

    total_error = np.sum(w_edge * abs(label[edge[:,0]]-label[edge[:,1]]))

    return total_error

def measure_consistency(model, data, edge, w_edge):
    """         
        Computes the consistency score, 1 - sum (|yi_hat - yj_hat| * Wij) / sum (Wij)

        Args:
            model: Trained model
            data: Features of the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs

        Return:
            consistency score: Consistency score on the data
    """
    
    y_hat = model.predict(data)
    
    total_error = measure_error(y_hat, edge, w_edge)
    num_similar_pairs = len(edge)
    
    consistency_score = 1 - total_error / num_similar_pairs
    return consistency_score

def generate_sim_matrix(data, similarity_matrix, similarity_params):
    """         
        Obtains kNN-based/threshold-based similarity matrix and similar pairs.
            kNN-based: Considers (xi, xj) as a similar pair if xi is one of xj's nearest neighbors or vice versa.
            threshold-based: Considers (xi, xj) as a similar pair if their distance is smaller than a threshold.

        Args:
            data: Features of the data
            similarity_matrix: "knn" or "threshold" based similarity matrix
            similarity_params: Hyperparameters for similarity matrix

        Return:
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
    """

    n, d = data.shape
    w_sim = np.zeros((n, n))

    params = falconn.get_default_parameters(n, d)
    params.lsh_family = falconn.LSHFamily.Hyperplane

    params.k = similarity_params["num_hash"]
    params.l = similarity_params["num_table"]

    table = falconn.LSHIndex(params)
    table.setup(data)
    qo = table.construct_query_object()
    
    edge = []
    for i in range(data.shape[0]):
        if similarity_matrix == "knn":
            indices = qo.find_k_nearest_neighbors(data[i], similarity_params["k"]+1)
        elif similarity_matrix == "threshold":
            indices = qo.find_near_neighbors(data[i], similarity_params["threshold"]**2)

        distances = np.squeeze(euclidean_distances([data[i]], data[indices]))
        w_sim[i, indices] = np.exp(-1 * similarity_params["theta"] * distances)
        w_sim[indices, i] = np.exp(-1 * similarity_params["theta"] * distances)
        w_sim[i][i] = 0

    temp = np.argwhere(w_sim > 0)
    edge = np.squeeze(temp[np.argwhere(temp[:, 0] < temp[:, 1])])
    w_edge = w_sim[edge[:,0], edge[:,1]]

    return w_sim, edge, w_edge

def performance_func(model, data, label, edge, w_edge):
    """         
        Returns model's accuracy and consistency score on the data.

        Args:
            model: Trained model
            data: Features of the data
            label: Labels of the data
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs

        Return:
            accuracy: Model's accuracy on the data
            consistency score: Consistency score on the data
    """

    accuracy = model.score(data, label)
    consistency_score = measure_consistency(model, data, edge, w_edge)
    
    return accuracy, consistency_score