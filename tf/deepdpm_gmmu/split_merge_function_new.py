import tensorflow as tf
import numpy as np
import pandas as pd
from GMMU.gmmu_cavi_stable_new import *
from deepdpm_gmmu.niw_prior import *

DATA_TYPE = 'float64'

#the same as the one in gmm_uniform
def calculate_log_det(var):
    """
    Calculate log determinant in log space, to prevent overflow errors.
    args:
        var:            tf.Tensor (1, k, d, d)
    return:
        log_det:        tf.Tensor (k, 1)
    """
    log_det = []
    n_components=var.shape[1]

    for k in range(n_components):
        evals, evecs = tf.linalg.eig(var[0, k])

        log_det.append(tf.reduce_sum(tf.math.log(tf.math.real(evals))))
    log_det = tf.convert_to_tensor(log_det)
    return tf.expand_dims(log_det, -1)


def check_inv(var):
    result = tf.linalg.pinv(var)
    if tf.reduce_sum(tf.cast(tf.math.is_nan(result), tf.int32)) > 0:
        return 1
    return 0

def acceptpro_split_hs(x, X_split, X_sub1, X_sub2, \
                       r_all, r_c1, r_c2, \
                       m_0, beta_0, w_0, nu_0, alpha=1.0):
    '''
    acceptance probability of split one cluster into two sub-clusters.
    args:
        x:          tf.Tensor (n, d)
        X_split:    tf.Tensor (n_cluster_k, d)
        X_sub1:     tf.Tensor (n_sub1, d)
        X_sub2:     tf.Tensor (n_sub2, d)
        r_all:      tf.Tensor (n, )
        r_c1:       tf.Tensor (n, )
        r_c2:       tf.Tensor (n, )
        m_0:        tf.Tensor (1, d)
        beta_0:     tf.Tensor (1,)
        w_0:      tf.Tensor (d, d)
        nu_0:       tf.Tensor (1,)
        alpha:      float32
    returns:
        acceptpro:  tf.Tensor (1,)
    '''
    N = X_split.shape[0]
    n_sub1 = X_sub1.shape[0]
    n_sub2 = X_sub2.shape[0]

    alpha = tf.cast(alpha, x.dtype)

    prob_c_original = tf.reduce_sum(tf.math.log(tf.range(1, N, dtype=x.dtype)))
    prob_c_split1 = tf.reduce_sum(tf.math.log(tf.range(1, n_sub1, dtype=x.dtype)))
    prob_c_split2 = tf.reduce_sum(tf.math.log(tf.range(1, n_sub2, dtype=x.dtype)))

    #log likelihood of cluster_original
    cr = r_all  # (n, )
    cNk = tf.reduce_sum(cr, axis=0) #(1, )
    c_beta = update_beta(beta_0, cNk) #(1,)
    c_nu = update_nu(nu_0, cNk) #(1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x) #(d, d)

    cluster_split_log_prob = log_marginal_likelihood2(X_split, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster1
    cr = r_c1  # (n, )
    cNk = tf.reduce_sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    sub1_log_prob = log_marginal_likelihood2(X_sub1, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster2
    cr = r_c2  # (n,)
    cNk = tf.reduce_sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    sub2_log_prob = log_marginal_likelihood2(X_sub2, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)
    
    log_acceptpro = tf.math.log(alpha)+prob_c_split1+prob_c_split2-prob_c_original \
                    + sub1_log_prob+sub2_log_prob-cluster_split_log_prob \
                    + tf.cast((n_sub1+n_sub2-2)*tf.math.log(2.), dtype=x.dtype)
    if np.isnan(log_acceptpro):
        if np.isnan(sub1_log_prob):
            print('sub1_log_prob',sub1_log_prob)
        if np.isnan(sub2_log_prob):
            print('sub2_log_prob',sub2_log_prob)
        if np.isnan(cluster_split_log_prob):
            print('cluster_split_log_prob',cluster_split_log_prob)
    #if tf.exp(log_acceptpro)==0:
    #    print('log hs', log_acceptpro)

    return tf.exp(log_acceptpro)


def acceptpro_merg_hm(x, X_merge, X_c1, X_c2, \
                       r_all, r_c1, r_c2, \
                       m_0, beta_0, w_0, nu_0, alpha=1):
    '''
    acceptance probability of merge two clusters into one cluster.
    args:
        x:          tf.Tensor (n, d)
        X_merge:    tf.Tensor (n_merge, d)
        X_c1:       tf.Tensor (n_c1, d)
        X_c2:       tf.Tensor (n_c2, d)
        r_all:      tf.Tensor (n,)
        r_c1:       tf.Tensor (n,)
        r_c2:       tf.Tensor (n,)
        m_0:        tf.Tensor (1, d)
        beta_0:     tf.Tensor (1,)
        w_0:      tf.Tensor (d, d)
        nu_0:       tf.Tensor (1,)
        alpha:      float32
    returns:
        acceptpro:  tf.Tensor (1,)
    '''
    N = X_merge.shape[0]
    n_c1 = X_c1.shape[0]
    n_c2 = X_c2.shape[0]

    alpha = tf.cast(alpha, x.dtype)

    prob_c_merge = tf.reduce_sum(tf.math.log(tf.range(1, N, dtype=x.dtype)))
    prob_c1 = tf.reduce_sum(tf.math.log(tf.range(1, n_c1, dtype=x.dtype)))
    prob_c2 = tf.reduce_sum(tf.math.log(tf.range(1, n_c2, dtype=x.dtype)))

    #log likelihood of cluster_merge
    cr = r_all  # (n, )
    cNk = tf.reduce_sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    cluster_merge_log_prob = log_marginal_likelihood2(X_merge, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster1
    cr = r_c1  # (n, )
    cNk = tf.reduce_sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    c1_log_prob = log_marginal_likelihood2(X_c1, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster2
    cr = r_c2  # (n, )
    cNk = tf.reduce_sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    c2_log_prob = log_marginal_likelihood2(X_c2, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    log_acceptpro = -tf.math.log(alpha)-prob_c1-prob_c2+prob_c_merge \
                    -c1_log_prob-c2_log_prob+cluster_merge_log_prob\
                    - tf.cast((n_c1+n_c2-2)*tf.math.log(2.), dtype=x.dtype)
    if np.isnan(log_acceptpro):
        if np.isnan(c1_log_prob):
            print('c1_log_prob',c1_log_prob)
        if np.isnan(c2_log_prob):
            print('c2_log_prob',c2_log_prob)
        if np.isnan(cluster_merge_log_prob):
            print('cluster_merge_log_prob',cluster_merge_log_prob)
    #if tf.exp(log_acceptpro)==0:
    #    print('log hm', log_acceptpro)
    return tf.exp(log_acceptpro)


