from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_auc_score
from aif360.metrics import ClassificationMetric

def measure_violations(label, edge):
    """         
        Computes the number volations in the data.

        Args:
            label: Labels of the data
            edge: Indices of similar pairs

        Return:
            num_violations: The number of violations
    """

    num_violations = int(np.sum(abs(label[edge[:,0]]-label[edge[:,1]])))

    return num_violations

def measure_consistency(model, data, w_sim, edge):
    """         
        Computes the consistency score, 1 - sum (|yi_hat - yj_hat| * Wij) / sum (Wij)

        Args:
            model: Trained model
            data: Features o1f the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs

        Return:
            consistency score: Consistency score on the data
    """
    
    y_hat = model.predict(data)
    
    num_violations = measure_violations(y_hat, edge)
    num_similar_pairs = len(edge)
    
    consistency_score = 1 - num_violations / num_similar_pairs
    return consistency_score

def similarity_threshold(data, threshold):
    """         
        Obtains threshold-based similarity matrix and similar pairs.
        threshold-based similarity matrix: Considers (xi, xj) as a similar pair if their distance is smaller than a threhold.

        Args:
            data: Features of the data
            threshold: threshold used to define similarity

        Return:
            w_sim: Similarity matrix
            edge: Indices of similar pairs
    """
    
    sim = euclidean_distances(data, data)
    w_sim = sim < threshold
    edge = []
    for i in range(data.shape[0]):
        w_sim[i][i] = 0
        
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            if w_sim[i][j]:
                edge.append((i, j))
                
    return w_sim, np.array(edge)

def similarity_knn(data, k):
    """         
        Obtains kNN-based similarity matrix and similar pairs.
        kNN-based similarity matrix: Considers (xi, xj) as a similar pair if xi is one of xj's nearest neighbors or vice versa.

        Args:
            data: Features of the data
            k: k used to define similarity

        Return:
            w_sim: Similarity matrix
            edge: Indices of similar pairs
    """

    distances = euclidean_distances(data, data)

    w_sim = np.zeros((data.shape[0], data.shape[0]))
    edge = []
    for i in range(data.shape[0]):
        index = np.argsort(distances[i])[:k+1]
        w_sim[i, index] = 1
        w_sim[index, i] = 1
        w_sim[i][i] = 0
    
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            if w_sim[i][j]:
                edge.append((i, j))

    return w_sim, np.array(edge)

def performance_func(model, data, label, w_sim, edge):
    """         
        Returns model's accuracy and consistency score on the data.

        Args:
            model: Trained model
            data: Features of the data
            label: Labels of the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs

        Return:
            accuracy: Model's accuracy on the data
            consistency score: Consistency score on the data
    """

    accuracy = model.score(data, label)
    consistency_score = measure_consistency(model, data, w_sim, edge)
    
    return accuracy, consistency_score