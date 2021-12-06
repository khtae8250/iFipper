from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

def init_cluster_info(valuei, i, valuej, j, cluster_info):
    if (valuei != valuej) and (1e-7 < valuei < 1-1e-7):
        if valuei not in cluster_info:
            cluster_info[valuei] = dict()
            cluster_info[valuei][valuej] = [j]
        else:
            if valuej not in cluster_info[valuei]:
                cluster_info[valuei][valuej] = [j]
            else:
                cluster_info[valuei][valuej].append(j)

    return cluster_info

def init_cluster(label, original_label, edge):
    cluster_info, cluster_nodes, cluster_nodes_num = dict(), dict(), dict()
    for i in range(len(label)):
        value = label[i]
        if value not in cluster_nodes:
            cluster_nodes[value] = [i]
        else:
            cluster_nodes[value].append(i)

        if (1e-7 < value < 1-1e-7):
            if value not in cluster_nodes_num:
                cluster_nodes_num[value] = (1-2*original_label[i])
            else:
                cluster_nodes_num[value] += (1-2*original_label[i])

    for (i, j) in edge:
        valuei, valuej = label[i], label[j]

        cluster_info = init_cluster_info(valuei, i, valuej, j, cluster_info)
        cluster_info = init_cluster_info(valuej, j, valuei, i, cluster_info)

    return cluster_info, cluster_nodes, cluster_nodes_num

def update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes):
    
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

    if alpha_ not in cluster_nodes:
        cluster_nodes[alpha_] = cluster_nodes[alpha]
    else:
        cluster_nodes[alpha_].extend(cluster_nodes[alpha])

    del cluster_info[alpha]
    del cluster_nodes_num[alpha]
    del cluster_nodes[alpha]

    return cluster_info, cluster_nodes_num, cluster_nodes

def get_zero_cluster(cluster_nodes_num):
    alpha = 0
    for value in cluster_nodes_num:
        if cluster_nodes_num[value] == 0:
            alpha = value
            break

    return alpha
    
def get_nonzero_two_clusters(cluster_info):
    value1 = 0
    for value in cluster_info:
        if value1 != 0:
            value2 = value
            break
        else:
            value1 = value

    alpha, beta = min(value1, value2), max(value1, value2)
    return alpha, beta

def get_cluster_info(alpha, cluster_info):

    k, U = 0, 0
    ak, ak1= 0, 1

    for connected_node in cluster_info[alpha]:
        U += len(cluster_info[alpha][connected_node])
        if connected_node < alpha:
            k += len(cluster_info[alpha][connected_node])

    return ak, ak1, k, U

def get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num):

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

def transform_with_one_cluster(alpha, cluster_info, cluster_nodes_num, cluster_nodes):
    ak, ak1, k, U = get_cluster_info(alpha, cluster_info)
    if 2*k - U <= 0:
        alpha_ = ak1
    else:
        alpha_ = ak

    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes 


def transform_with_two_clusters(alpha, beta, cluster_info, cluster_nodes_num, cluster_nodes):

    ak, ak1, N_alpha, k, U, bl, bl1, N_beta, l, V, E = get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num)
    X, Y = N_alpha / N_beta, ((2*k-U-E)*N_beta - (2*l-V+E)*N_alpha)/N_beta

    if X < 0 and Y <= 0:
        if X + 1 <=0:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta))
        else:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=5)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    elif X < 0 and Y > 0:
        if X + 1 >= 0: 
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl))
        else:
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl), -(N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=5)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    elif X > 0 and Y <= 0:
        epsilon_alpha = min(ak1-alpha, (N_beta/N_alpha)*(beta-bl), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=5)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    elif X > 0 and Y > 0:
        epsilon_alpha = min(alpha-ak, (N_beta/N_alpha)*(bl1-beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=5)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)
    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(beta, beta_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes