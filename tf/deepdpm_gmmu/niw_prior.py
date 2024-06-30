import tensorflow as tf
import numpy as np
import pandas as pd

def prior_niw(x, mu):
    """
    Return the prior parameters of niw distribution
    input:
       x:                tf.Tensor (n, d)
       mu:               tf.Tensor (1, k, d)
    output:
        m_0:             tf.Tensor (1, d)
        beta_0:          tf.Tensor (1,)
        psi_0:           tf.Tensor (d, d)
        nu_0:            tf.Tensor (1,)
    """
    K = mu.shape[1]
    D = x.shape[-1]
    if len(x.shape) == 2:
        # (n, d) --> (n, 1, d)
        x = tf.expand_dims(x, 1)
    m_0 = tf.reduce_mean(mu, axis=1)
    m_0 = tf.cast(m_0, dtype=tf.double)
    beta_0 = tf.constant(0.001, shape=[1], dtype=tf.double)

    # w_0=psi_0^-1
    x_mean = tf. reduce_mean(x, axis=0, keepdims=True)
    psi_0 = tf.linalg.diag(tf.squeeze(tf.reduce_sum((x - x_mean)**2, axis=0)/(K**(1/D))))
    psi_0 = tf.cast(psi_0, dtype=tf.double)

    nu_0 = tf.constant(D+2, shape=[1], dtype=tf.double)

    return m_0, beta_0, psi_0, nu_0


def calculating_Nk(soft_clustering):
    """
        input:
        soft_clustering:  tf.Tensor (n, k+1, 1)
        output:
        Nk:      tf.Tensor (k,1)
    """
    #exclude pi_0
    responsibility = soft_clustering[:, 1:, :]
    responsibility = responsibility / tf.reduce_sum(responsibility, axis=1, keepdims=True)  # (n, k, 1)
    Nk = tf.reduce_sum(responsibility, axis=0)  # (k, 1)
    return Nk

def calculating_responsibility(soft_clustering):
    """
        input:
        soft_clustering:  tf.Tensor (n, k+1, 1)
        output:
        responsibility:   tf.Tensor (n, k, 1)
    """
    #exclude pi_0
    responsibility = soft_clustering[:, 1:, :]
    responsibility = responsibility / tf.reduce_sum(responsibility, axis=1, keepdims=True)  # (n, k, 1)
    return responsibility


def update_beta(beta_0, Nk):
    """
        Updtate lambda_beta
        beta_o + Nk
    input:
    Nk:               tf.Tensor (k,)
    beta_0:           tf.Tensor (1,)
    output:
    lambda_beta:      tf.Tensor (k,)
    """
    lambda_beta = beta_0 + Nk
    return lambda_beta


def update_nu(nu_0, Nk):
    """
        Update lambda_nu
        nu_o + Nk
    input:
    Nk:              tf.Tensor (k,)
    nu_0:            tf.Tensor (1,)
    output:
    lambda_nu:       tf.Tensor (k,)
    """
    lambda_nu = nu_0 + Nk
    return lambda_nu


def update_m(responsibility, lambda_beta, m_0, beta_0, x):
    """
    Update lambda_m
    (m_o.T * beta_o + sum_{n=1}^{N}(E_{q_{z}} I(z_{n}=i)x_{n})) / lambda_beta
    input:
    responsibility:   tf.Tensor (n,)
    lambda_beta:      tf.Tensor (1,)
    m_0:              tf.Tensor (1, d)
    beta_0:           tf.Tensor (1,)
    x:                tf.Tensor (n, d)
    output:
    lambda_m:         tf.Tensor (1, d)
    """
    #Rnk * xn (k,d)
    lambda_m = tf.matmul(tf.expand_dims(responsibility, axis=0), x)
    lambda_m = lambda_m + beta_0*m_0
    lambda_m = lambda_m/lambda_beta

    return lambda_m



def update_w(responsibility, w_o, beta_o, m_o, x):
    '''
    return the updated w
    :param responsibility: (n, )
    :param w_o: (d, d)
    :param beta_o: (1,)
    :param m_o: (1, d)
    :param x: (n, d)
    :return:
    '''
    d = x.shape[-1]
    Nk = tf.reduce_sum(responsibility, axis=0) #shape:([])
    xbar = tf.matmul(tf.expand_dims(responsibility, axis=0), x)/ Nk #(d,)
    x_xbar = x - xbar  # (n,d)
    snk = tf.matmul(tf.expand_dims(x_xbar, axis=-1), tf.expand_dims(x_xbar, axis=-2)) # (n,d,d)
    Sk=tf.reduce_sum(tf.tile(tf.reshape(responsibility, [-1, 1, 1]), [1, d, d]) * snk, axis=0) / Nk
    #print(responsibility)
    inv_w_o = tf.linalg.inv(w_o)
    NkSk = Nk*Sk
    e1 = beta_o*Nk/(beta_o+Nk)
    e2 = tf.matmul(tf.expand_dims(xbar-m_o, axis=-1), tf.expand_dims(xbar-m_o, axis=-2))
    #print(inv_w_o, NkSk, e1*e2)
    lambda_w=tf.linalg.inv(inv_w_o + NkSk + e1*e2+ \
                           1e-6 * tf.eye(inv_w_o.shape[1], dtype=inv_w_o.dtype))
    return lambda_w


def update_w_inv(responsibility, w_o, beta_o, m_o, x):
    '''
    return the updated w_inv
    :param responsibility: (n, )
    :param w_o: (d, d)
    :param beta_o: (1,)
    :param m_o: (1, d)
    :param x: (n, d)
    :return:
    '''
    d = x.shape[-1]
    Nk = tf.reduce_sum(responsibility, axis=0) #shape:([])
    xbar = tf.matmul(tf.expand_dims(responsibility, axis=0), x)/ Nk #(d,)
    x_xbar = x - xbar  # (n,d)
    snk = tf.matmul(tf.expand_dims(x_xbar, axis=-1), tf.expand_dims(x_xbar, axis=-2)) # (n,d,d)
    Sk=tf.reduce_sum(tf.tile(tf.reshape(responsibility, [-1, 1, 1]), [1, d, d]) * snk, axis=0) / Nk
    #print(responsibility)
    inv_w_o = tf.linalg.inv(w_o)
    NkSk = Nk*Sk
    e1 = beta_o*Nk/(beta_o+Nk)
    e2 = tf.matmul(tf.expand_dims(xbar-m_o, axis=-1), tf.expand_dims(xbar-m_o, axis=-2))
    #print(inv_w_o, NkSk, e1*e2)
    lambda_w_inv=inv_w_o + NkSk + e1*e2
    return lambda_w_inv


def multi_t_density(x, mu, Lambda, nu):
    """
    calculate the pdf of multivariate t distribution
    input:
    x:              tf.Tensor (1, d)
    mu:             tf.Tensor (1, d)
    Lambda:         tf.Tensor (d, d)
    nu:             tf.Tensor (1,)
    output:
    pdf:            tf.Tensor (1,)
    """
    D = Lambda.shape[-1]
    log_pdf = tf.math.lgamma((nu+D)/2) - tf.math.lgamma(nu/2)
    log_pdf = log_pdf + 0.5 * (tf.linalg.logdet(Lambda)) - D/2*tf.math.log(np.pi*nu)
    delta = tf.matmul(tf.matmul((x - mu), Lambda), tf.transpose(x - mu))
    log_pdf = log_pdf - (D+nu)/2 * tf.math.log(1 + delta**2/nu)
    pdf = tf.exp(log_pdf)
    return pdf

def log_multi_t_density(x, mu, Lambda, nu):
    """
    calculate the pdf of multivariate t distribution
    input:
    x:              tf.Tensor (1, d)
    mu:             tf.Tensor (1, d)
    Lambda:         tf.Tensor (d, d)
    nu:             tf.Tensor (1,)
    output:
    pdf:            tf.Tensor (1,)
    """
    D = Lambda.shape[-1]
    log_pdf = tf.math.lgamma((nu+D)/2) - tf.math.lgamma(nu/2)
    log_pdf = log_pdf + 0.5 * (tf.linalg.logdet(Lambda)) - D/2*tf.math.log(np.pi*nu)
    delta = tf.matmul(tf.matmul((x - mu), Lambda), tf.transpose(x - mu))
    log_pdf = log_pdf - (D+nu)/2 * tf.math.log(1 + delta**2/nu)
    return log_pdf

def multi_log_gamma_function(input, p):
    """
    input: tf.Tensor (1,)
    p:     tf.Tensor (1,)
    """
    C = tf.math.log(np.pi) * p * (p-1) /4
    C = tf.cast(C, dtype=input.dtype)
    for i in range(p):
        C = C + tf.math.lgamma(input-i/2)
    return C

def log_marginal_likelihood(x, beta_0, w_0, nu_0, lambda_beta, lambda_w, lambda_nu):
    """
    calculate the pdf of multivariate t distribution
    input:
    x:              tf.Tensor (n, d)
    beta_0:         tf.Tensor (1,)
    w_0:          tf.Tensor (d, d)
    nu_0:           tf.Tensor (1,)
    lambda_beta:    tf.Tensor (1, )
    lambda_w:     tf.Tensor (d, d)
    lambda_nu:      tf.Tensor (1, )
    output:
    lml:            tf.Tensor (1,)
    """
    N = x.shape[0]
    D = x.shape[-1]
    lml = - N*D/2*tf.math.log(np.pi)
    lml = tf.cast(lml, dtype=x.dtype)
    beta_0 = tf.cast(beta_0, dtype=x.dtype)
    w_0 = tf.cast(w_0, dtype=x.dtype)
    nu_0 = tf.cast(nu_0, dtype=x.dtype)
    lambda_beta = tf.cast(lambda_beta, dtype=x.dtype)
    lambda_w = tf.cast(lambda_w, dtype=x.dtype)
    lambda_nu = tf.cast(lambda_nu, dtype=x.dtype)

    lml = lml + multi_log_gamma_function(lambda_nu/2, D) - multi_log_gamma_function(nu_0/2, D)
    #print('1',lml)
    lml = lml + nu_0/2*tf.linalg.logdet(tf.linalg.inv(w_0)) - \
        lambda_nu/2*tf.linalg.logdet(tf.linalg.inv(lambda_w))
    #print('2',lml)
    #print(tf.linalg.det(lambda_w), tf.math.log(tf.linalg.det(lambda_w)))
    lml = lml + (D/2)*(tf.math.log(beta_0)-tf.math.log(lambda_beta))
    #print('3',lml)
    return lml


def log_marginal_likelihood2(x, beta_0, w_0, nu_0, lambda_beta, lambda_w_inv, lambda_nu):
    """
    calculating by the inverse of w, which can help reduce the det value of w ( otherwise, may cause the inf)
    calculate the pdf of multivariate t distribution
    input:
    x:              tf.Tensor (n, d)
    beta_0:         tf.Tensor (1,)
    w_0:          tf.Tensor (d, d)
    nu_0:           tf.Tensor (1,)
    lambda_beta:    tf.Tensor (1, )
    lambda_w:     tf.Tensor (d, d)
    lambda_nu:      tf.Tensor (1, )
    output:
    lml:            tf.Tensor (1,)
    """
    N = x.shape[0]
    D = x.shape[-1]
    lml = - N*D/2*tf.math.log(np.pi)
    lml = tf.cast(lml, dtype=x.dtype)
    beta_0 = tf.cast(beta_0, dtype=x.dtype)
    w_0 = tf.cast(w_0, dtype=x.dtype)
    nu_0 = tf.cast(nu_0, dtype=x.dtype)
    lambda_beta = tf.cast(lambda_beta, dtype=x.dtype)
    lambda_w_inv = tf.cast(lambda_w_inv, dtype=x.dtype)
    lambda_nu = tf.cast(lambda_nu, dtype=x.dtype)

    lml = lml + multi_log_gamma_function(lambda_nu/2, D) - multi_log_gamma_function(nu_0/2, D)
    lml = lml + nu_0/2*tf.math.log(tf.linalg.det(tf.linalg.inv(w_0))) - \
        lambda_nu/2*tf.linalg.logdet(lambda_w_inv)
    lml = lml + (D/2)*(tf.math.log(beta_0)-tf.math.log(lambda_beta))
    return lml











