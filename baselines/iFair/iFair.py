"""
Implementation of the ICDE 2019 paper
iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making
url: https://ieeexplore.ieee.org/document/8731591
citation:
@inproceedings{DBLP:conf/icde/LahotiGW19,
  author    = {Preethi Lahoti and
               Krishna P. Gummadi and
               Gerhard Weikum},
  title     = {iFair: Learning Individually Fair Data Representations for Algorithmic
               Decision Making},
  booktitle = {35th {IEEE} International Conference on Data Engineering, {ICDE} 2019,
               Macao, China, April 8-11, 2019},
  pages     = {1334--1345},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/ICDE.2019.00121},
  doi       = {10.1109/ICDE.2019.00121},
  timestamp = {Wed, 16 Oct 2019 14:14:56 +0200},
  biburl    = {https://dblp.org/rec/conf/icde/LahotiGW19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
__author__: Preethi Lahoti
__email__: plahoti@mpi-inf.mpg.de
"""
import numpy as np
from lowrank_helpers import iFair as ifair_func
from lowrank_helpers import predict as ifair_predict
import sklearn.metrics.pairwise as pairwise

from scipy.optimize import minimize, fmin_l_bfgs_b

class iFair:

    def __init__(self, k=2, A_x=1e-2, A_z=1.0, max_iter=10000, nb_restarts=5):
        self.k = k
        self.A_x = A_x
        self.A_z = A_z
        self.max_iter = max_iter
        self.nb_restarts = nb_restarts
        self.opt_params = None
        self.opt_result = None

    def fit(self, X_train, dataset=None):
        """
        Learn the model using the training data. iFair.py._func
        :param X:     Training data. Expects last column of the matrix X to be the protected attribute.
        """
        
        ##if dataset object is not passed, assume that there is only 1 protected attribute and it is the last column of X
        if dataset:
            D_X_F = pairwise.euclidean_distances(X_train[:, dataset.nonsensitive_column_indices], X_train[:, dataset.nonsensitive_column_indices])
            l = len(dataset.nonsensitive_column_indices)
        else:
            D_X_F = pairwise.euclidean_distances(X_train[:, :-1],
                                                 X_train[:, :-1])
            l = X_train.shape[1] - 1

        P = X_train.shape[1]
        min_obj = None
        opt_params = None
        for i in range(self.nb_restarts):
            ifair_func.iters = 0
            params = np.random.uniform(size= P * 2 + self.k + P * self.k)

            # setting protected column weights to epsilon
            ## assumes that the column indices from l through P are protected and appear at the end
            for i in range(l, P, 1):
                params[i] = 0.0001

            bnd = []
            for i, k2 in enumerate(params):
                if i < P * 2 or i >= P * 2 + self.k:
                    bnd.append((None, None))
                else:
                    bnd.append((0, 1))

            opt_result = fmin_l_bfgs_b(ifair_func, x0=params,
                                args=(X_train, D_X_F, self.k, self.A_x, self.A_z, 0),
                                approx_grad=True,
                                bounds=bnd,
                                maxiter=self.max_iter,
                                maxfun=self.max_iter,
                                epsilon=1e-3)

        if (min_obj is None) or (opt_result[1] < min_obj):
            min_obj = opt_result[1]
            opt_params = opt_result[0]
            
        self.opt_params = opt_params

    def transform(self, X, dataset = None):
        X_hat = ifair_predict(self.opt_params, X, k=self.k)
        return X_hat

    def fit_transform(self, X_train, dataset=None):
        """
        Learns the model from the training data and returns the data in the new space.
        :param X:   Training data.
        :return:    Training data in the new space.
        """
        # print('Fitting and transforming...')
        self.fit(X_train, dataset)
        return self.transform(X_train)