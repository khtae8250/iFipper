from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy


def init_cluster(optimal_label, label, edge):
    """         
        Initializes cluster information.

        Args: 
            optimal_label: Optimal solution for the LP problem
            label: Labels of the data
            edge: Indices of similar pairs

        Return: 
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster
    """

    cluster_info, cluster_nodes, cluster_nodes_num = dict(), dict(), dict()
    for i in range(len(optimal_label)):
        value = optimal_label[i]
        if value not in cluster_nodes:
            cluster_nodes[value] = [i]
        else:
            cluster_nodes[value].append(i)

        if (1e-7 < value < 1-1e-7):
            if value not in cluster_nodes_num:
                cluster_nodes_num[value] = (1-2*label[i])
            else:
                cluster_nodes_num[value] += (1-2*label[i])

    for (i, j) in edge:
        valuei, valuej = optimal_label[i], optimal_label[j]
        if valuei != valuej:
            cluster_info = init_cluster_info(valuei, i, valuej, j, cluster_info)
            cluster_info = init_cluster_info(valuej, j, valuei, i, cluster_info)

    return cluster_info, cluster_nodes, cluster_nodes_num

def init_cluster_info(valuei, i, valuej, j, cluster_info):
    """         
        Constructs cluster_info for a similar pair.

        Args: 
            valuei: Optimal label for yi, which is non-0/1 value
            i: Index of yi
            valuej: Optimal label for yj, which is non-0/1 value
            j: Index of yj
            cluster_info: Contains information about the connected clusters for each cluster

        Return:
            cluster_info: Updated cluster_info using a similar pair
    """

    if (1e-7 < valuei < 1-1e-7):
        if valuei not in cluster_info:
            cluster_info[valuei] = dict()
            cluster_info[valuei][valuej] = [j]
        else:
            if valuej not in cluster_info[valuei]:
                cluster_info[valuei][valuej] = [j]
            else:
                cluster_info[valuei][valuej].append(j)

    return cluster_info

def transform_with_one_cluster(alpha, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        For an alpha-cluster with N_alpha=0, it converts alpha to either a_k or a_k+1 while maintaining an optimal solution (Lemma 2.1 in the paper).

        Args: 
            alpha: Initial value of the alpha-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    ak, ak1, k, U = get_cluster_info(alpha, cluster_info)
    if 2*k - U <= 0:
        alpha_ = ak1
    else:
        alpha_ = ak

    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes 

def get_zero_cluster(cluster_nodes_num):
    """         
        Obtains an alpha-cluster with N_alpha=0 in the solution.

        Args: 
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            alpha: Value of the alpha-cluster
    """

    alpha = 0
    for value in cluster_nodes_num:
        if cluster_nodes_num[value] == 0:
            alpha = value
            break

    return alpha

def get_cluster_info(alpha, cluster_info):
    """         
        Obtains information about nodes connected to the alpha-cluster.

        Args: 
            alpha: Value of the alpha-cluster
            cluster_info: Contains information about the connected clusters for each cluster

        Return: 
            ak: The largest value among nodes connected to the alpha-cluster whose values are less than the alpha value
            ak1: The smallest value among nodes connected to the alpha-cluster whose values are greater than the alpha value
            k: Number of nodes connected to the alpha-cluster whose values are less than the alpha value
            U: Number of nodes connected to the alpha-clutser
    """

    k, U = 0, 0
    ak, ak1= 0, 1
    for connected_node in cluster_info[alpha]:
        U += len(cluster_info[alpha][connected_node])
        if connected_node < alpha:
            k += len(cluster_info[alpha][connected_node])

        if ak < connected_node and connected_node < alpha:
            ak = connected_node
        elif ak1 > connected_node and connected_node > alpha:
            ak1 = connected_node
            
    return ak, ak1, k, U

def transform_with_two_clusters(alpha, beta, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        For an alpha-cluster with N_alpha!=0 and a beta-cluster with N_beta!=0, it converts (alpha, beta) to one of the five cases in Lemma 2.2 in the paper.

        Args: 
            alpha: Initial value of the alpha=cluster
            beta: Initial value of the beta-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    ak, ak1, N_alpha, k, U, bl, bl1, N_beta, l, V, E = get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num)
    X, Y = N_alpha / N_beta, ((2*k-U-E)*N_beta - (2*l-V+E)*N_alpha)/N_beta
    
    if X < 0 and Y <= 0:
        if X + 1 <=0:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta))
        else:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=4)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=4)

    elif X < 0 and Y > 0:
        if X + 1 >= 0: 
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl))
        else:
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl), -(N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=4)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=4)

    elif X > 0 and Y <= 0:
        epsilon_alpha = min(ak1-alpha, (N_beta/N_alpha)*(beta-bl), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=4)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=4)

    elif X > 0 and Y > 0:
        epsilon_alpha = min(alpha-ak, (N_beta/N_alpha)*(bl1-beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=4)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=4)

    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)
    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(beta, beta_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes

def get_nonzero_two_clusters(cluster_info):
    """         
        Obtains an alpha-cluster and a beta-cluster in the solution

        Args: 
            cluster_info: Contains information about the connected clusters for each cluster

        Return: 
            alpha: Value of the alpha-cluster
            beta: Value of the beta-cluster
    """

    value1 = 0
    for value in cluster_info:
        if value1 != 0:
            value2 = value
            break
        else:
            value1 = value

    alpha, beta = min(value1, value2), max(value1, value2)
    return alpha, beta

def get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num):
    """         
        Obtains information about nodes connected to the alpha-cluster and beta-cluster.

        Args: 
            alpha: Value of the alpha-cluster
            beta: Initial value of the beta-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            ak: The largest value among nodes connected to the alpha-cluster whose values are less than the alpha value and are not beta value
            ak1: The smallest value among nodes connected to the alpha-cluster whose values are greater than the alpha value and are not beta value
            N_alpha: N_alpha value of the alpha-cluster
            k: Number of nodes connected to the alpha-cluster whose values are less than the alpha value and are not beta value
            U: Number of nodes connected to the alpha-clutser whose values are not beta value
            bl: The largest value among nodes connected to the beta-cluster whose values are less than the beta value and are not alpha value
            bl1: The smallest value among nodes connected to the beta-cluster whose values are greater than the beta value and are not alpha value
            N_beta: N_beta value of the beta-cluster
            l: Number of nodes connected to the beta-cluster whose values are less than the beta value and are not alpha value
            V: Number of nodes connected to the beta-clutser whose values are not alpha value
            E: Number of edges between the two clusters
    """

    k, U, l, V, E = 0, 0, 0, 0, 0
    ak, ak1, bl, bl1 = 0, 1, 0, 1
    N_alpha, N_beta = cluster_nodes_num[alpha], cluster_nodes_num[beta]

    for connected_node in cluster_info[alpha]:
        U += len(cluster_info[alpha][connected_node])
        if connected_node < alpha:
            k += len(cluster_info[alpha][connected_node])

        if connected_node != beta:
            if ak < connected_node and connected_node < alpha:
                ak = connected_node
            elif ak1 > connected_node and connected_node > alpha:
                ak1 = connected_node
        else:
            E = len(cluster_info[alpha][connected_node])
            U = U - E
        
    for connected_node in cluster_info[beta]:
        V += len(cluster_info[beta][connected_node])
        if connected_node < beta:
            l += len(cluster_info[beta][connected_node])

        if connected_node != alpha:
            if bl < connected_node and connected_node < beta:
                bl = connected_node
            elif bl1 > connected_node and connected_node > beta:
                bl1 = connected_node
        else:
            l = V - E
            V = V - E

    return ak, ak1, N_alpha, k, U, bl, bl1, N_beta, l, V, E

def update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        Updates related cluster information after converting alpha to alpha_ in the solution.

        Args: 
            alpha: Initial value of the cluster
            alpha_: Converted value of the cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    for connected_node in cluster_info[alpha]:
        if (1e-7 < connected_node < 1-1e-7):
            if alpha_ not in cluster_info[connected_node]:
                cluster_info[connected_node][alpha_] = cluster_info[connected_node][alpha]

            del cluster_info[connected_node][alpha]

    if (1e-7 < alpha_< 1-1e-7):
        if alpha_ not in cluster_info:
            cluster_info[alpha_] = cluster_info[alpha]
            cluster_nodes_num[alpha_] = cluster_nodes_num[alpha]
        else:
            cluster_nodes_num[alpha_] += cluster_nodes_num[alpha]
            for connected_node in cluster_info[alpha]:
                if connected_node not in cluster_info[alpha_]:
                    cluster_info[alpha_][connected_node] = cluster_info[alpha][connected_node]
                else:
                    cluster_info[alpha_][connected_node].extend(cluster_info[alpha][connected_node])

        if alpha_ in cluster_info[alpha_]:
            del cluster_info[alpha_][alpha_]

    if alpha_ not in cluster_nodes:
        cluster_nodes[alpha_] = cluster_nodes[alpha]
    else:
        cluster_nodes[alpha_].extend(cluster_nodes[alpha])

    del cluster_info[alpha]
    del cluster_nodes_num[alpha]
    del cluster_nodes[alpha]

    return cluster_info, cluster_nodes_num, cluster_nodes
