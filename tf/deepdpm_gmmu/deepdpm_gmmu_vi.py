import tensorflow as tf
import numpy as np
import pandas as pd
from split_merge_function import *
import math
import random
from GMMU.gmmu_cavi import *
from GMM import *


DATA_TYPE = 'float64'

class deepdpm(tf.keras.models.Model):
    '''
    a structure updates the number of clusters automatically
    that consists of clustering part (GMM with uniform),
    sub-clustering part (MLP or GMM) and split and merge part.
    '''

    def __init__(self, n_components, init_params="kmeans", max_iter_dpm=500, tol_dpm=2,
                 subcluster_reinitial=0.8, threshold=1e+1, random_state=None, alpha=1, unif_a=None, unif_b=None):
        '''
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               tf.Tensor (n, d)
            mu:              tf.Tensor (k, d)
            var:             tf.Tensor (k, d, d)
            pi:              tf.Tensor (k, )
        '''
        super(deepdpm, self).__init__()
        # parameters of deepdpm
        self.max_iter_dpm = max_iter_dpm
        self.tol_dpm = tol_dpm
        self.subcluster_reinitial = subcluster_reinitial
        self.alpha = alpha

        # parameters of clustering net
        self.threshold = threshold
        self.random_state = random_state

        self.n_components = n_components

        self.init_params = init_params
        self.unif_a = unif_a
        self.unif_b = unif_b

        # clusteringnet initialization
        self.clusteringnet = CAVI_GMMU(n_cluster = self.n_components, a_o = self.unif_a, b_o = self.unif_b,  \
                 threshold=self.threshold, init_param=self.init_params, random_state=self.random_state)

        # sub-clustering net initialization
        self.subcluteringnet = {}
        for i in range(1, self.n_components + 1):
            self.subcluteringnet[str(i)] = GMM(n_components=2)  # initialization

    def _init_params(self):
        pass

    def check_size(self, x):
        if len(x.shape) == 2:
            # (n, d) --> (n, 1, d)
            x = tf.expand_dims(x, 1)
        return x

    def __check_inv(self):
        result = tf.linalg.inv(self.var)
        if tf.reduce_sum(tf.cast(tf.math.is_nan(result), tf.int32)) > 0:
            return 1
        return 0

    def fit(self, x):
        """
        Fits model to the data.
        args:
            x:          np.array (n, d)
        options:


        """
        self.n_features = x.shape[1]
        self.n_sample = x.shape[0]
        self._init_params()
        k_old = self.n_components
        x = x.astype(DATA_TYPE)
        x_subclusterkey_old = np.random.randint(1, k_old, size=self.n_sample)
        subclusternet_old = self.subcluteringnet
        clusteringnet_old = self.clusteringnet
        iter = 0
        stable_num = 0
        while (iter <= self.max_iter_dpm) and (stable_num < self.tol_dpm):
            # fit clustering net
            k_new = k_old
            clusteringnet_old.fit(x)

            # get prior parameters of niw distribution
            alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = clusteringnet_old.parameters_o()
            # responsibility/soft assignment (n,k)
            r_cluster = clusteringnet_old.soft_assignment()


            # fit each subclustering net and split at the same time
            hardcluster_label = clusteringnet_old.hard_assignment()
            x_subclusterkey_new = np.zeros(self.n_sample)
            subclusternet_new = {}
            for i in range(1, k_old + 1):  # uniform has no subclustering net so i start from 1
                subcluster_i_mask = hardcluster_label == i
                x_i = tf.boolean_mask(x, mask=subcluster_i_mask)
                subcluster_key = tf.boolean_mask(tf.cast(x_subclusterkey_old, tf.int32), mask=subcluster_i_mask)

                # find the subclustering net in the dictionary, because the key and the output of GMM are not same
                key_list, _, key_count = tf.unique_with_counts(subcluster_key)
                if tf.reduce_max(key_count) > tf.cast(self.subcluster_reinitial * x_i.shape[0], tf.int32):
                    x_subclusterkey_new[subcluster_i_mask] = i
                    if key_list[tf.argmax(key_count)].numpy() != 0:
                        subclusternet_new[str(i)] = subclusternet_old[str(key_list[tf.argmax(key_count)].numpy())]
                    else:
                        x_subclusterkey_new[subcluster_i_mask] = i
                        subclusternet_new[str(i)] = GMM(n_components=2)
                else:  # lots of samples from different cluster, reintialize the subcluster
                    x_subclusterkey_new[subcluster_i_mask] = i
                    subclusternet_new[str(i)] = GMM(n_components=2)

                subcluster_i = subclusternet_new[str(i)]
                # if x_i contains 1 or less sample, skip to fit subclustering net
                if x_i.shape[0] <= 1:
                    continue
                # fit the subcluster nets
                subcluster_i.fit(x_i)


                # split step
                ##calculating the acceptance probability
                subcluster_i_hard_label = subcluster_i.hard_clustering(x_i)
                subcluster_i1_mask = np.array(subcluster_i_hard_label == 0)
                subcluster_i2_mask = np.array(subcluster_i_hard_label == 1)
                x_i1 = tf.boolean_mask(x_i, mask=subcluster_i1_mask)
                x_i2 = tf.boolean_mask(x_i, mask=subcluster_i2_mask)
                if x_i1.shape[0] == 0 or x_i2.shape[0] == 0:
                    continue
                r_all = r_cluster[:, i]  # (n, )
                r_sub = tf.expand_dims(r_all, axis=-1) * subcluster_i.soft_clustering(x)  # (n, 2)
                r_c1 = r_sub[:, 0]  # (n, )
                r_c2 = r_sub[:, 1]  # (n, )
                hs = acceptpro_split_hs(x, x_i, x_i1, x_i2, r_all, r_c1, r_c2, \
                                        m_o, beta_o, w_o, nu_o, self.alpha)
                print('hs', hs)
                if tf.random.uniform(shape=[], minval=0., maxval=1) < tf.cast(hs, tf.float32):
                    k_new = k_new + 1
                    # new subclustering net
                    x_subclusterkey_new_clusternew = x_subclusterkey_new[subcluster_i_mask]
                    x_subclusterkey_new_clusternew[subcluster_i2_mask] = k_new
                    x_subclusterkey_new[subcluster_i_mask] = x_subclusterkey_new_clusternew
                    subclusternet_new[str(i)] = GMM(n_components=2)
                    subclusternet_new[str(k_new)] = GMM(n_components=2)

            # update GMM with uniform
            if k_new != k_old:
                clusteringnet_new = CAVI_GMMU(n_cluster=k_new, a_o=self.unif_a, b_o=self.unif_b, \
                          threshold=self.threshold, init_param=self.init_params, random_state=self.random_state)
                clusteringnet_new.fit(x)
            else:
                clusteringnet_new = clusteringnet_old

            # merge
            ##get new parameters for the merge part
            k_new2 = k_new  # recorde the number of merging
            # get prior parameters of niw distribution
            alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = clusteringnet_new.parameters_o()
            lambda_alpha, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o = clusteringnet_new.parameters()
            # responsibility/soft assignment (n,k)
            softcluster_label = clusteringnet_new.soft_assignment() #tensor
            hardcluster_label = clusteringnet_new.hard_assignment() #numpy
            #print('cluster', hardcluster_label)
            #print('prior', lambda_alpha/tf.reduce_sum(lambda_alpha))


            dm = self.distance_matrix(lambda_m, k_new, self.n_features)
            merged_list = []
            nan_cluster_list = [] #the clusters that contain no point
            if k_new > 1:
                n_pair = min(3, k_new - 1)
                merge_pair = tf.argsort(dm)[:, 1: (n_pair + 1)]
                for i, pairs in enumerate(merge_pair):
                    if (i + 1) in list(np.array(merged_list).flat) or (i + 1) in nan_cluster_list:
                        # plus 1 because considering uniform pi_0
                        continue
                    mask_i = hardcluster_label == (i + 1)  # plus 1 because considering uniform pi_0
                    X_c1 = tf.boolean_mask(x, mask=mask_i)
                    if X_c1.shape[0] == 0:
                        nan_cluster_list.append(i + 1)
                        k_new2 = k_new2 - 1
                        continue
                    for j in range(n_pair):
                        if (pairs[j].numpy() + 1) in list(np.array(merged_list).flat) \
                                or (pairs[j].numpy() + 1) in nan_cluster_list:
                            # plus 1 because considering uniform pi_0
                            continue
                        pair = pairs[j].numpy()
                        mask_j = hardcluster_label == (pair + 1)  # plus 1 because considering uniform pi_0
                        X_c2 = tf.boolean_mask(x, mask=mask_j)
                        X_merge = tf.concat([X_c1, X_c2], 0)

                        if X_c2.shape[0] == 0 :
                            nan_cluster_list.append(pair + 1)
                            k_new2 = k_new2 - 1
                            continue

                        r = softcluster_label[:, 1:]#(n,k)
                        r_c1 = r[:, i] #(n,)
                        r_c2 = r[:, pair] #(n,)
                        r_merge = r_c1 + r_c2 #(n,)

                        hm = acceptpro_merg_hm(x, X_merge, X_c1, X_c2, \
                                               r_merge, r_c1, r_c2, \
                                               m_o, beta_o, w_o, nu_o, self.alpha)
                        print('hm', str([i+1, pair+1]), hm)
                        if (X_c1.shape[0] == 0) or (X_c2.shape[0] == 0) or \
                                (tf.random.uniform(shape=[], minval=0., maxval=1) < tf.cast(hm, tf.float32)):
                            merged_list.append([i + 1, pair + 1])
                            k_new2 = k_new2 - 1
                            x_subclusterkey_new[mask_j] = i+1
                            subclusternet_new[str(i+1)] = GMM(n_components=2)

            # merge step: drop old subcluster nets and add a new one; update the subcluster dictionary
            print('merged_list: ', merged_list)
            print('nan_cluster_list: ', nan_cluster_list)

            print("k_old", k_old, "k_new", k_new, "k_new2", k_new2)
            # if the number of components is stable, iteration will terminate.
            if k_new2 == k_old:
                stable_num = stable_num + 1
            else:
                stable_num = 0
            k_old = k_new2

            # update GMM with uniform and other
            if k_new2 != k_new:
                clusteringnet_old = CAVI_GMMU(n_cluster=k_new2, a_o=self.unif_a, b_o=self.unif_b, \
                                              threshold=self.threshold, init_param=self.init_params,
                                              random_state=self.random_state)
            else:
                clusteringnet_old = clusteringnet_new

            x_subclusterkey_old = x_subclusterkey_new
            subclusternet_old = subclusternet_new

            iter = iter + 1

        # have determined n_components
        self.n_components = k_old
        self.clusteringnet = clusteringnet_old
        self.clusteringnet.fit(x)

    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        tf.Tensor (n, k)
            (or)
            y:          tf.LongTensor (n)
        """
        return self.clusteringnet.predict(x, probs)

    def hard_clustering(self):
        """
        Returns Hard Clustering probabilities of training data.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            y:          tf.LongTensor (n)
        """
        return self.clusteringnet.hard_assignment()

    def soft_clustering(self):
        """
        Returns Soft Clustering probabilities of training data.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            y:          tf.LongTensor (n, k+1)
        """
        return self.clusteringnet.soft_assignment()

    def num_components(self):
        '''
        return the number of components
        :return: self.n_components
        '''
        return self.n_components

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            score:      tf.Tensor (n)
        """
        return self.clusteringnet.score_samples(x)

    def distance_matrix(self, t, n_components, n_feature):
        '''
        calculate the distance matrix of mu of each cluster
        args:
            t:                  tf.Tensor (1, k, d) or array (1, k, d)
            n_components:       int #k
            n_feature:          int #d
        return:
            distance_matrix:    tf.Tensor (k, k)
        '''
        t1 = tf.reshape(t, (1, n_components, n_feature))
        t2 = tf.reshape(t, (n_components, 1, n_feature))
        distance_matrix = tf.norm(t1 - t2, ord='euclidean', axis=2)
        return distance_matrix

    def cluster_prior(self):
        lambda_alpha, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o = self.clusteringnet.parameters()
        return lambda_alpha/tf.reduce_sum(lambda_alpha)
