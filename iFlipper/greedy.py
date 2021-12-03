from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from utils import measure_violations
    
def Greedy(label, m, w_sim, edge):
    """         
        Repeatedly flips labels that reduce the number of violations the most.
        
        Args: 
            label: Labels of the data
            m: The violations limit
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    flipped_label = copy.deepcopy(label)

    num_violations = measure_violations(flipped_label, edge)
    prev_violations = num_violations
    while num_violations > m:
        max_indices, improvement_arr = [], []
        total_improvement = 0
        
        for j in range(len(flipped_label)):
            prev_y_diff = (flipped_label != flipped_label[j])

            improvement = np.sum(w_sim[j] * (2 * prev_y_diff - 1))
            improvement_arr.append(improvement)

        sorted_index = np.argsort(improvement_arr)[::-1][:1]
        for index in sorted_index:
            if improvement_arr[index] <= 0:
                break

            total_improvement += improvement_arr[index]
            max_indices.append(index)
            if total_improvement >= (num_violations - m):
                break

        flipped_label[max_indices] = 1-flipped_label[max_indices]
        num_violations = measure_violations(flipped_label, edge)

        # there is no more violations reduction
        if prev_violations <= num_violations:
            print("no more reduction")
            break
        else:
            prev_violations = num_violations
            
    return flipped_label