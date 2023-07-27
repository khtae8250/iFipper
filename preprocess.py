from aif360.datasets import AdultDataset, CompasDataset, GermanDataset, BankDataset, StandardDataset
from sklearn.preprocessing import StandardScaler
import math
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data

import numpy as np

from iFlipper.utils import generate_sim_matrix, generate_original_sim_matrix

def get_dataset(opt):
    similarity_params = dict()
    dataset_type = opt.dataset

    if dataset_type == "COMPAS":
        dataset = CompasDataset(label_name='two_year_recid', favorable_classes=[0], 
                                protected_attribute_names=['sex'], privileged_classes=[['Female']], 
                                categorical_features=['age_cat', 'c_charge_degree', 'c_charge_desc', 'race'], 
                                features_to_keep=['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'two_year_recid'], 
                                features_to_drop=[], na_values=[])
        
        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 1, 10, 0.05
        similarity_params["k"], similarity_params["threshold"] = 20, 3

        num_plot = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.001, 0]
        m_list = [15000, 1500, 150]
        
    elif dataset_type == "AdultCensus":
        dataset = AdultDataset(label_name='income-per-year', favorable_classes=['>50K', '>50K.'], 
                        protected_attribute_names=['sex'], privileged_classes=[['Male']], 
                        categorical_features=['race', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country'], 
                        features_to_keep=[], features_to_drop=['fnlwgt'], na_values=['?'])

        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 10, 50, 0.1
        similarity_params["k"], similarity_params["threshold"]  = 20, 3

        num_plot = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
        m_list = [10000, 1000, 100]
        
    elif dataset_type == "Credit":
        def label_processing(df):
            credit_map = {1.0 : 1.0, 2.0 : 0.0}
            status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                        'A92': 'female', 'A95': 'female'}
            df['credit'] = df['credit'].replace(credit_map)
            df['sex'] = df['personal_status'].replace(status_map)    
            return df
        
        dataset = GermanDataset(custom_preprocessing = label_processing, 
                                metadata={'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}], 'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}, {1.0: 'Old', 0.0: 'Young'}]})

        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 1, 10, 0.05
        similarity_params["k"], similarity_params["threshold"]  = 20, 7
        
        num_plot = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        m_list = [1000, 100, 0]

    # This is only for time estimation
    elif dataset_type == "Synthetic":
        X, Y = generate_synthetic_data(opt)
        x_train, x_test, x_val = X[:int(opt.num_train*len(Y))], X[int(opt.num_train*len(Y)):int(opt.num_train*len(Y))+int(opt.num_test*len(Y))], X[int(opt.num_train*len(Y))+int(opt.num_test*len(Y)):]
        y_train, y_test, y_val = Y[:int(opt.num_train*len(Y))], Y[int(opt.num_train*len(Y)):int(opt.num_train*len(Y))+int(opt.num_test*len(Y))], Y[int(opt.num_train*len(Y))+int(opt.num_test*len(Y)):]
        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 1, 10, 0.05
        similarity_params["k"], similarity_params["threshold"] = 20, 3

        size_arr = [100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]

        dset = lambda x:1
        dset.x_train = x_train
        dset.x_test = x_test
        dset.x_val = x_val

        dset.y_train = y_train
        dset.y_test = y_test
        dset.y_val = y_val

        plots = dict()
        plots["size_arr"] = size_arr
        return similarity_params, dset, plots


    if dataset_type in ["COMPAS", "AdultCensus", "Credit"]:
        scaler = StandardScaler()
        train, test, val = dataset.split([opt.num_train, opt.num_train+opt.num_test], shuffle=True, seed=opt.seed)

        train.features = scaler.fit_transform(train.features)
        test.features = scaler.fit_transform(test.features)
        val.features = scaler.fit_transform(val.features)
        index = train.feature_names.index(opt.protected)

        # remove sensitive features
        x_train = np.delete(train.features, index, axis=1)
        x_test = np.delete(test.features, index, axis=1)
        x_val = np.delete(val.features, index, axis=1)

        y_train = train.labels.ravel()
        y_test = test.labels.ravel()
        y_val = val.labels.ravel()

        # keep sensitive column for PFR
        train_sensitive = np.reshape(train.features[:, index], (-1, 1))
        test_sensitive = np.reshape(test.features[:, index], (-1, 1))
        val_sensitive = np.reshape(val.features[:, index], (-1, 1))

        # For iFair
        x_train_with_sensitive = np.concatenate((x_train, train_sensitive), axis = 1)
        x_test_with_sensitive = np.concatenate((x_test, test_sensitive), axis = 1)
        x_val_with_sensitive = np.concatenate((x_val, val_sensitive), axis = 1)

    # For return
    dset =  lambda x:1
    dset.train = train
    dset.test = test
    dset.val = val
    dset.index = index

    dset.x_train = x_train
    dset.x_test = x_test
    dset.x_val = x_val

    dset.y_train = y_train
    dset.y_test = y_test
    dset.y_val = y_val

    dset.x_train_with_sensitive = x_train_with_sensitive
    dset.x_test_with_sensitive = x_test_with_sensitive
    dset.x_val_with_sensitive = x_val_with_sensitive

    plots = dict()
    plots['num_plot'] = num_plot
    plots['m_list'] = m_list

    if opt.verbose:
        print("Dataset: %s" % dataset_type)
        print("Number of training data: %d, Number of test data: %d, Number of validation data: %d\n" % (x_train.shape[0], x_test.shape[0], x_val.shape[0]))

    return similarity_params, dset, plots

def get_sim_matrix(opt, x, similarity_params):
    """
        Synonom for iFlipper.utils.generate_sim_matrix
    """
    similarity_matrix = opt.similarity_matrix
    return generate_sim_matrix(x, similarity_matrix, similarity_params)


def generate_synthetic_data(opt, n_samples1=2000, n_samples2=1000):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).

        Args:
            n_samples1: number of samples for positive sensitive class
            n_samples2: number of samples for negative sensitive class
        Return:
            X: generated X
            y: generated y

    """
    np.random.seed(opt.seed)
    seed(opt.seed)

    # n_samples = 1000 # generate these many data points per class
    disc_factor = math.pi / 4.0 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label, n_samples):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1, n_samples1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, 0, n_samples2) # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = np.arange(0,n_samples1+n_samples2)
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    
    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    return X,y
