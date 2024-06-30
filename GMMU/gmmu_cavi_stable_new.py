'''
CAVI
'''
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from numpy import random
import numbers
from math import pi
import matplotlib.pyplot as plt
import time
from collections import Counter
# from GMMU.util import *
# from GMMU.util import *
from sklearn.mixture import GaussianMixture


import tensorflow as tf
import numpy as np
from numpy import random
import numbers
from math import pi
import matplotlib.pyplot as plt

from numpy import sin, cos
from scipy.special import gamma, multigammaln, gammaln
from scipy.stats import wishart
from numpy.random import multivariate_normal
from scipy.stats import multivariate_normal as norm
from numpy.linalg import inv, det, cholesky
import scipy
import matplotlib as mpl
import imageio, os



def dirichlet_expectation(alpha):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.math.subtract(tf.math.digamma(tf.add(alpha, np.finfo(np.float64).eps)),
                       tf.math.digamma(tf.reduce_sum(alpha)))


def dirichlet_expectation_k(alpha, k):
    """
    Dirichlet expectation computation
    \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
    """
    return tf.subtract(tf.math.digamma(tf.add(alpha[k], np.finfo(np.float64).eps)),
                       tf.math.digamma(tf.reduce_sum(alpha)))


def log_beta_function(x):
    """
    Log beta function
    ln(\gamma(x)) - ln(\gamma(\sum_{i=1}^{N}(x_{i}))
    """
    return tf.subtract(
        tf.reduce_sum(tf.math.lgamma(tf.add(x, np.finfo(np.float64).eps))),
        tf.math.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float64).eps))))


def softmax(x, axis=1):
    """
    Softmax computation
    e^{x} / sum_{i=1}^{K}(e^x_{i})
    """
    if len(x.shape)==1:
        x=tf.expand_dims(x, axis=0)
    return tf.math.divide(tf.add(tf.math.exp(tf.subtract(x, tf.reduce_max(x, axis=axis, keepdims=True))),
                         np.finfo(np.float64).eps),
                  tf.reduce_sum(
                      tf.add(tf.math.exp(tf.subtract(x, tf.reduce_max(x, axis=axis, keepdims=True))),
                             np.finfo(np.float64).eps), axis=axis, keepdims=True))



def multilgamma(a, D, D_t):
    """
    ln multigamma Tensorflow implementation
    """
    res = tf.math.multiply(tf.math.multiply(D_t, tf.math.multiply(tf.subtract(D_t, 1),
                                                   tf.cast(0.25,
                                                           dtype=tf.float64))),
                      tf.math.log(tf.cast(np.pi, dtype=tf.float64)))
    res += tf.reduce_sum(tf.math.lgamma([tf.subtract(a, tf.div(
        tf.subtract(tf.cast(j, dtype=tf.float64),
                    tf.cast(1., dtype=tf.float64)),
        tf.cast(2., dtype=tf.float64))) for j in range(1, D + 1)]), axis=0)
    return res


def log_(x):
    return tf.math.log(tf.add(x, np.finfo(np.float64).eps))


def generate_random_positive_matrix(D):
    """
    Generate a random semidefinite positive matrix
    :param D: Dimension
    :return: DxD matrix
    """
    aux = random.rand(D, D)
    return np.dot(aux, aux.transpose())


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def draw_ellipse(mu, cov, num_sd=1):
    # as I understand it -
    # diagonalise the cov matrix
    # use eigenvalues for radii
    # use eigenvector rotation from x axis for rotation
    x = mu[0]  # x-position of the center
    y = mu[1]  # y-position of the center
    cov = cov * num_sd  # show within num_sd standard deviations
    lam, V = np.linalg.eig(cov)  # eigenvalues and vectors
    t_rot = np.arctan(V[1, 0] / V[0, 0])
    a, b = lam[0], lam[1]
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    # 2-D rotation matrix

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    return x + Ell_rot[0, :], y + Ell_rot[1, :]

cols = [\
        '#8159a4',
        '#60c4bf',
        '#f19c39',
        '#cb5763',
        '#6e8dd7',
        ]

def plot_GMM(X, mu, lam, pi, centres, covs, K, title, cols=cols, savefigpath=False):
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'kx', alpha=0.2)

    legend = ['Datapoints']

    for k in range(K):
        cov = lam[k]
        x_ell, y_ell = draw_ellipse(mu[k], cov)
        plt.plot(x_ell, y_ell, cols[k])
        legend.append('pi=%.2f, var1=%.2f, var2=%.2f, cov=%.2f' % (pi[k], cov[0, 0], cov[1, 1], cov[1, 0]))
        # for whatever reason lots of the ellipses are very long and narrow, why?

    # Plotting the ellipses for the GMM that generated the data
    for i in range(centres.shape[0]):
        x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
        plt.plot(x_true_ell, y_true_ell, 'g--')
        legend.append(
            'Data generation GMM %d, var1=%.2f, var2=%.2f, cov=%.2f' % (i, covs[i][0, 0], covs[i][1, 1], covs[i][1, 0]))

    plt.title(title)
    plt.plot(mu[:,0], mu[:,1], 'ro')
    if isinstance(savefigpath, str):
        plt.savefig(savefigpath)
    else:
        plt.legend(legend)
        plt.show()


def multi_t_density(x, mu, Lambda, nu):
    """
    calculate the pdf of multivariate t distribution
    input:
    x:              tf.Tensor (1, d)
    mu:             tf.Tensor (k, d)
    Lambda:         tf.Tensor (k, d, d)
    nu:             tf.Tensor (k,)
    output:
    pdf:            tf.Tensor (k,)
    """
    D = Lambda.shape[-1]
    log_pdf = tf.math.lgamma((nu+D)/2) - tf.math.lgamma(nu/2)
    log_pdf = log_pdf + 0.5 * (tf.math.log(tf.linalg.det(Lambda))) - D/2*tf.math.log(np.pi*nu)
    delta = tf.matmul(tf.matmul(tf.expand_dims(x - mu, axis=-2), Lambda), tf.expand_dims(x - mu, axis=-1))
    log_pdf = log_pdf - (D+nu)/2 * tf.math.log(1 + tf.reshape(delta, [-1])**2/nu)
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
    log_pdf = log_pdf + 0.5 * (tf.math.log(tf.linalg.det(Lambda))) - D/2*tf.math.log(np.pi*nu)
    delta = tf.matmul(tf.matmul((x - mu), Lambda), tf.transpose(x - mu))
    log_pdf = log_pdf - (D+nu)/2 * tf.math.log(1 + delta**2/nu)
    return log_pdf

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class CAVI_GMMU(tf.keras.models.Model):
    def __init__(self, n_cluster, a_o = None, b_o = None, u_filter = False, u_filter_rate=0.0025, \
                 threshold=1e+1, init_param="kmeans", random_state=None, gif_display = False, \
                 true_mean=None, true_var=None, reg_covar = 1e-6, DATA_TYPE = 'float64', weights_init=None, \
                 means_init=None, precisions_init=None):
        """
        Parameter Inizatiloasion
        """
        super(CAVI_GMMU, self).__init__()
        self.k = n_cluster
        self.a_o = a_o
        self.b_o = b_o
        self.u_filter = u_filter
        self.u_filter_rate = u_filter_rate
        self.threshold = threshold
        self.random_state = random_state
        self.gif_display = gif_display
        self.true_mean = true_mean
        self.true_var = true_var
        self.init_param = init_param
        self.DATA_TYPE = DATA_TYPE
        self.reg_covar = reg_covar
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        assert self.init_param in ["kmeans", "random", "gmm", "self_setting"]



    def _init_param(self, x):       # x.shape=[114,32]
        """
        remaining: define self.d
        """
        chi2_dist = tfp.distributions.Chi2(df = self.d)     # d=32
        # Get the quantile of chi2
        self.u_chi2_quantile = chi2_dist.quantile(1-self.u_filter_rate)# u_filter_rate = 0.025

        #Priors for Uniform
        if self.u_filter:
            if self.a_o is None:
                self.a_o = tf.cast(tf.reduce_min(x, axis=0), dtype=self.DATA_TYPE)
                self.a_o = tf.convert_to_tensor(self.a_o, dtype=self.DATA_TYPE)
                # self.a_o=[32]
            if self.b_o is None:
                self.b_o = tf.cast(tf.reduce_max(x, axis=0), dtype=self.DATA_TYPE)
                self.b_o = tf.convert_to_tensor(self.b_o, dtype=self.DATA_TYPE)

            w = generate_random_positive_matrix(self.d) # self.d=32, w=(32,32)
            self.u_b_a_o = 1/ (tf.cast((2.0 * np.pi)**(-self.d/2), dtype=self.DATA_TYPE) * \
                           tf.cast(tf.linalg.det(w)**(-1/2), dtype=self.DATA_TYPE) * \
                           tf.cast(tf.exp(-0.5*self.u_chi2_quantile), dtype=self.DATA_TYPE))

            self.lambda_u_b_a = self.add_weight(name="lambda_u_b_a",
                                                 shape=self.u_b_a_o.shape,
                                                 dtype=self.DATA_TYPE,
                                                 initializer=tf.initializers.Constant(self.u_b_a_o),
                                                 trainable=True)
        else:
            if self.a_o is None:
                self.a_o = tf.cast(tf.reduce_min(x, axis=0), dtype=self.DATA_TYPE)
            self.a_o = tf.convert_to_tensor(self.a_o, dtype=self.DATA_TYPE)

            if self.b_o is None:
                self.b_o = tf.cast(tf.reduce_max(x, axis=0), dtype=self.DATA_TYPE)
            self.b_o = tf.convert_to_tensor(self.b_o, dtype=self.DATA_TYPE)

            self.u_b_a_o = tf.math.cumprod(self.b_o - self.a_o)[-1]

            self.lambda_u_b_a = self.add_weight(name="lambda_u_b_a",
                                                 shape=self.u_b_a_o.shape,
                                                 dtype=self.DATA_TYPE,
                                                 initializer=tf.initializers.Constant(self.u_b_a_o),
                                                 trainable=True)

        # Priors
        alpha_o = np.array([1.0] * (self.k+1)) #add 1 for uniform, alpha_o.shape=(14,)
        nu_o = np.array([float(self.d + 100)])
        w_o = generate_random_positive_matrix(self.d)   # w_o.shape=(32,32)
        m_o = np.zeros(self.d)#np.mean(x, axis=0)
        beta_o = np.array([0.7])

        # Variational parameters intialization
        lambda_pi_var = np.ones(shape=self.k + 1) #add 1 for uniform
        lambda_beta_var = np.ones(shape=self.k)*0.7
        lambda_nu_var = np.ones(shape=self.k)*2. + self.d
        if self.init_param in ["kmeans", "gmm", "self_setting"]:
            mat = []
            for i in range(self.k):
                xk = x[tf.argmax(self.lambda_phi, axis=1) == (i+1), :]
                mat.append(
                    np.linalg.inv((xk - self.lambda_m[i].numpy()).T @ (xk - self.lambda_m[i].numpy()) \
                                  / (xk.shape[0] - 1) + self.reg_covar * tf.eye(xk.shape[1], dtype=self.DATA_TYPE)) \
                    / lambda_nu_var[i])
            lambda_w_var = np.array(mat)
        else:
            mat = []
            for i in range(self.k):
                mat.append(generate_random_positive_matrix(self.d))
            lambda_w_var = np.array(mat)

        self.lambda_pi = self.add_weight(name="lambda_pi",
                                             shape=lambda_pi_var.shape,
                                             dtype=self.DATA_TYPE,
                                             initializer=tf.initializers.Constant(lambda_pi_var),
                                             trainable=True)

        self.lambda_beta = self.add_weight(name="lambda_beta",
                                               shape=lambda_beta_var.shape,
                                               dtype=self.DATA_TYPE,
                                               initializer=tf.initializers.Constant(lambda_beta_var),
                                               trainable=True)

        self.lambda_nu = self.add_weight(name="lambda_nu",
                                             shape=lambda_nu_var.shape,
                                             dtype=self.DATA_TYPE,
                                             initializer=tf.initializers.Constant(lambda_nu_var),
                                             trainable=True)

        self.lambda_w = self.add_weight(name="lambda_w",
                                            shape=lambda_w_var.shape,
                                            dtype=self.DATA_TYPE,
                                            initializer=tf.initializers.Constant(lambda_w_var),
                                            trainable=True)


        self.alpha_o = tf.convert_to_tensor(alpha_o, dtype=self.DATA_TYPE)
        self.nu_o = tf.convert_to_tensor(nu_o, dtype=self.DATA_TYPE)
        self.w_o = tf.convert_to_tensor(w_o, dtype=self.DATA_TYPE)
        self.m_o = tf.convert_to_tensor(m_o, dtype=self.DATA_TYPE)
        self.beta_o = tf.convert_to_tensor(beta_o, dtype=self.DATA_TYPE)

    def update_lambda_pi(self, lambda_pi, Nk):
        lambda_pi.assign(self.alpha_o + Nk)

    def update_lambda_u_b_a(self, lambda_u_b_a, Nk, Sk):
        #if np.any(np.isnan(self.lambda_phi)):
        #    print('u_b_a_o and Nk[0]:',self.u_b_a_o,Nk[0])
        if self.u_filter:
            Sk_g = Sk[1:]
            v = tf.reduce_max(tf.linalg.det(Sk_g))
            self.u_b_a_o = 1 / (tf.cast((2.0 * np.pi) ** (-self.d / 2), dtype=self.DATA_TYPE) * \
                                tf.cast(v ** (-1 / 2), dtype=self.DATA_TYPE) * \
                                tf.cast(tf.exp(-0.5 * self.u_chi2_quantile), dtype=self.DATA_TYPE))
        if self.u_b_a_o - Nk[0] > 0:
            lambda_u_b_a.assign(self.u_b_a_o - Nk[0])
        else:
            lambda_u_b_a.assign(self.u_b_a_o)

    def update_lambda_beta(self, lambda_beta, Nk):
        #only use the gaussian part
        Nk_g = Nk[1:]
        lambda_beta.assign(self.beta_o + Nk_g)

    def update_lambda_nu(self, lambda_nu, Nk):
        # only use the gaussian part
        Nk_g = Nk[1:]
        lambda_nu.assign(self.nu_o + Nk_g)

    def update_lambda_m(self, lambda_m, Nk, xbar):
        # only use the gaussian part
        Nk_g = Nk[1:]
        xbar_g = xbar[1:]
        new_m = (self.beta_o*self.m_o + tf.tile(tf.reshape(Nk_g, [-1, 1]), [1, self.d])*xbar_g)/ \
                tf.tile(tf.reshape(self.lambda_beta, [-1, 1]), [1, self.d])
        lambda_m.assign(new_m)

    def update_lambda_w(self, lambda_w, Nk, Sk, xbar):
        # only use the gaussian part
        Nk_g = Nk[1:]
        xbar_g = xbar[1:]
        Sk_g = Sk[1:]
        K = self.k
        inv_w_o = tf.linalg.inv(self.w_o)
        for k in range(K):
            NkSk = Nk_g[k]*Sk_g[k]
            e1 = self.beta_o*Nk_g[k]/(self.beta_o+Nk_g[k])
            e2 = tf.matmul(tf.expand_dims(xbar_g[k]-self.m_o, axis=-1), tf.expand_dims(xbar_g[k]-self.m_o, axis=-2))
            lambda_w[k, :, :].assign(tf.linalg.inv(inv_w_o + NkSk + e1*e2 \
                                        + self.reg_covar * tf.eye(inv_w_o.shape[1], dtype=self.DATA_TYPE)))



    def update_lambda_phi(self, lambda_phi, xn):
        """
        Update lambda_phi
        softmax[dirichlet_expectation(lambda_pi) +
            lambda_m * lambda_nu * lambda_w^{-1} * x_{n} -
            1/2 * lambda_nu * lambda_w^{-1} * x_{n} * x_{n}.T -
            1/2 * lambda_beta^{-1} -
            lambda_nu * lambda_m.T * lambda_w^{-1} * lambda_m +
            D/2 * log(2) +
            1/2 * sum_{i=1}^{D}(\Psi(lambda_nu/2 + (1-i)/2)) -
            1/2 log(|lambda_w|)]
        """
        N = xn.shape[0]
        D = xn.shape[-1]
        #new_lambda_phi_n0 = tf.cast(tf.math.log(1.0),dtype=DATA_TYPE) - tf.math.digamma(self.lambda_u_b_a) + \
        #                    dirichlet_expectation_k(self.lambda_pi, 0) #tf.math.log(1.) is a hyperparamter
        new_lambda_phi_n0 = -tf.math.log(self.lambda_u_b_a) + \
                            dirichlet_expectation_k(self.lambda_pi, 0)
        # uniform part
        lambda_phi[:, 0].assign(tf.tile(tf.reshape(new_lambda_phi_n0, [1, ]), [N, ]))
        #gaussian part;try different D in psi calculation
        new_lambda_phi_nk = tf.reshape(dirichlet_expectation(self.lambda_pi)[1: ], [1, -1])\
        + tf.squeeze(tf.expand_dims(self.lambda_m, axis=-2) @ \
               (tf.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w @ tf.reshape(xn, [-1, 1, D, 1]))) \
        - tf.squeeze(tf.linalg.trace(((1 / 2) * tf.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w) @ \
               tf.matmul(tf.reshape(xn, [-1, 1, D, 1]), tf.reshape(xn, [-1, 1, 1, D])))) \
        - tf.reshape((D / 2) * (1 / self.lambda_beta), [1, -1]) \
        - tf.reshape((1 / 2) * tf.expand_dims(self.lambda_m, axis=-2) @ \
            (tf.reshape(self.lambda_nu, [-1, 1, 1]) * \
             self.lambda_w @ tf.expand_dims(self.lambda_m, axis=-1)), [1, -1]) \
        + (D / 2) * np.log(2.0) \
        + tf.reshape((1 / 2) * tf.math.reduce_sum( \
            [tf.math.digamma(self.lambda_nu / 2 + (1 - i) / 2) for
             i in range(D)], axis=0), [1, -1]) \
        + tf.reshape((1 / 2) * tf.linalg.logdet(self.lambda_w), [1, -1]) \
        - (D / 2) * np.log(2.0 * np.pi)
        lambda_phi[:, 1:].assign(tf.reshape(new_lambda_phi_nk,[N,-1]))

        lambda_phi.assign(softmax(lambda_phi))

    def update_lambda_phi2(self, lambda_phi, xn, Sk):
        """
        """
        Sk_g = Sk[1:]
        var = Sk_g 
        precision = tf.cast(tf.linalg.inv(var + self.reg_covar * tf.eye(var.shape[1], dtype=self.DATA_TYPE)), \
                            self.DATA_TYPE)
        N = xn.shape[0]
        D = xn.shape[-1]
        #new_lambda_phi_n0 = tf.cast(tf.math.log(1.0), dtype=DATA_TYPE) - tf.math.digamma(self.lambda_u_b_a) + \
        #                    dirichlet_expectation_k(self.lambda_pi, 0) #tf.math.log(1.) is a hyperparamter
        new_lambda_phi_n0 = -tf.math.log(self.lambda_u_b_a) + \
                            dirichlet_expectation_k(self.lambda_pi, 0)
        # uniform part
        lambda_phi[:, 0].assign(tf.tile(tf.reshape(new_lambda_phi_n0, [1, ]), [N, ]))
        #gaussian part
        new_lambda_phi_nk = tf.reshape(dirichlet_expectation(self.lambda_pi)[1: ], [1, -1])\
        + tf.squeeze(tf.expand_dims(self.lambda_m, axis=-2) @ \
               (precision @ tf.reshape(xn, [-1, 1, D, 1]))) \
        - tf.squeeze(tf.linalg.trace(((1 / 2) * precision) @ \
               tf.matmul(tf.reshape(xn, [-1, 1, D, 1]), tf.reshape(xn, [-1, 1, 1, D])))) \
        - tf.reshape((1 / 2) * tf.expand_dims(self.lambda_m, axis=-2) @ \
            (precision @ tf.expand_dims(self.lambda_m, axis=-1)), [1, -1]) \
        + tf.reshape((1 / 2) * tf.linalg.logdet(precision), [1, -1]) \
        - (D / 2) * np.log(2.0 * np.pi)
        lambda_phi[:, 1:].assign(tf.reshape(new_lambda_phi_nk,[N,-1]))

        lambda_phi.assign(softmax(lambda_phi))
        #if np.any(np.isnan(lambda_phi)):
            #print('###gmmu###there is nan in lambda_phi')
            #print('Sk: ',Sk)
            #print('precision',precision)
            #print('self.lambda_u_b_a',self.lambda_u_b_a)
            #print('self.lambda_pi',self.lambda_pi)
            #print('new_lambda_phi_n0: ',new_lambda_phi_n0)
            #print('new_lambda_phi_nk: ',new_lambda_phi_nk)
            #print('term1:',tf.reshape(dirichlet_expectation(self.lambda_pi)[1: ], [1, -1]))
            #print('term2:',tf.squeeze(tf.expand_dims(self.lambda_m, axis=-2) @ \
            #   (precision @ tf.reshape(xn, [-1, 1, D, 1]))))
            #print('term3:',tf.squeeze(tf.linalg.trace(((1 / 2) * precision) @ \
            #   tf.matmul(tf.reshape(xn, [-1, 1, D, 1]), tf.reshape(xn, [-1, 1, 1, D])))))
            #print('term4:',tf.reshape((1 / 2) * tf.expand_dims(self.lambda_m, axis=-2) @ \
            #(precision @ tf.expand_dims(self.lambda_m, axis=-1)), [1, -1]))
            #print('term5',tf.reshape((1 / 2) * tf.linalg.logdet(precision), [1, -1]))
            #print('after softmax',lambda_phi)



    def elbo(self, xn):
        """
        Evidence Lower Bound definition
        """
        # print("in")
        D = xn.shape[-1]

        e3 = tf.convert_to_tensor(0., dtype=self.DATA_TYPE)
        e2 = tf.convert_to_tensor(0., dtype=self.DATA_TYPE)
        h2 = tf.convert_to_tensor(0., dtype=self.DATA_TYPE)
        e1 = -log_beta_function(self.alpha_o) + tf.reduce_sum((self.alpha_o-1)*dirichlet_expectation(self.lambda_pi))
        h1 = log_beta_function(self.lambda_pi) - tf.reduce_sum((self.lambda_pi-1)*dirichlet_expectation(self.lambda_pi))

        logdet = tf.convert_to_tensor([tf.linalg.logdet(self.lambda_w[i, :, :]) for i in range(self.k)], dtype=self.DATA_TYPE)

        logDeltak = tf.cast(D * tf.math.log(2.0), dtype=self.DATA_TYPE) + logdet
        for i in range(1, D + 1):
            logDeltak = logDeltak + tf.math.digamma((self.lambda_nu + 1 - i) / 2.)
        logDeltak = tf.cast(logDeltak, dtype=self.DATA_TYPE)

        e2 = e2 + tf.reduce_sum(self.lambda_phi @ tf.reshape(dirichlet_expectation(self.lambda_pi), [-1, 1]))
        h2 = h2 - tf.linalg.trace(self.lambda_phi @ tf.math.log(tf.transpose(self.lambda_phi)))
        product = tf.squeeze(tf.expand_dims((tf.expand_dims(xn, axis=1) - self.lambda_m), axis=-2) @ self.lambda_w @ \
        tf.expand_dims((tf.expand_dims(xn, axis=1) - self.lambda_m), axis=-1))
        aux = logDeltak - tf.cast(D * tf.math.log(2. * np.pi), dtype=self.DATA_TYPE) \
              - self.lambda_nu * product - D / self.lambda_beta
        e3 = e3 + tf.reduce_sum(tf.cast(1 / 2, dtype=self.DATA_TYPE) * self.lambda_phi[:, 1:] * aux)
        #uniform part may influence the elbo dramatically
        e3 = e3 - tf.reduce_sum(self.lambda_phi[:, 0])*tf.math.log(self.lambda_u_b_a)
        # print(9)
        product = tf.convert_to_tensor([tf.matmul(tf.expand_dims(self.lambda_m[K,:]-self.m_o, axis=0),
                                 tf.matmul(self.lambda_w[K,:,:],
                    tf.expand_dims(self.lambda_m[K,:]-self.m_o, axis=-1))) for K in range(self.k)], dtype=self.DATA_TYPE)

        # print("8")
        traces = tf.convert_to_tensor([tf.linalg.trace(tf.matmul(tf.linalg.inv(self.w_o+\
                                                    self.reg_covar * tf.eye(self.w_o.shape[1], dtype=self.DATA_TYPE)),
                                                            self.lambda_w[K,:,:])) for K in range(self.k)])
        # print("7")
        h4 = -tf.reduce_sum(0.5*logDeltak + D/2*tf.math.log(self.lambda_beta/(2*np.pi))-tf.cast(D/2., dtype=self.DATA_TYPE))
        ###-lnB
        logB = self.lambda_nu / 2 * logdet + D * self.lambda_nu / 2 * tf.math.log(tf.cast(2., dtype=self.DATA_TYPE)) + D * (
                    D - 1) / 4 * tf.math.log(
            tf.cast(tf.constant(np.pi), dtype=self.DATA_TYPE))
        for i in range(1, D + 1):
            logB = logB + tf.math.lgamma((self.lambda_nu + 1 - i) / 2.)
        logB = tf.cast(logB, dtype=self.DATA_TYPE)
        ###part2: -lnB-(v-d-1)/2*E[ln|lambda_k|]+vD/2
        h5 = tf.reduce_sum(logB - (self.lambda_nu - D - 1) / 2 * logDeltak + self.lambda_nu * D / 2)
        h5 = tf.cast(h5, dtype=self.DATA_TYPE)

        ##E[ln p(mu,lambda)]
        ###part1
        e4 = tf.reduce_sum(0.5 * (D*tf.math.log(self.beta_o) + logDeltak - tf.cast(D * tf.math.log(
            2 * np.pi), dtype=self.DATA_TYPE) - self.beta_o * self.lambda_nu * product - D * self.beta_o / self.lambda_beta))
        e4 = tf.cast(e4, dtype=self.DATA_TYPE)
        ###-ln B_0
        logB = self.nu_o / 2 * tf.math.log(tf.linalg.det(self.w_o)) + \
               D * self.nu_o / 2 * tf.cast(tf.math.log(2.), dtype=self.DATA_TYPE) + D * (
                    D - 1) / 4 * tf.math.log(tf.cast(tf.constant(np.pi), dtype=self.DATA_TYPE))
        for i in range(1, D + 1):
            logB = logB + tf.math.lgamma((self.nu_o + 1 - i) / 2.)
        logB = tf.cast(logB, dtype=self.DATA_TYPE)
        ###part2 K*ln B_0 + ...
        e5 = tf.reduce_sum(-logB + (self.nu_o - D - 1) / 2 * logDeltak - self.lambda_nu / 2 * traces)
        e5 = tf.cast(e5, dtype=self.DATA_TYPE)
        # print("1")
        LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5
        #print(e1, e2, e3, e4, e5, h1, h2, h4, h5)
        return LB

    def fit(self, x, max_iter=100):     # [199,32]
        """
        """
        tf.random.set_seed(self.random_state)
        x_tf = tf.convert_to_tensor(x, dtype=self.DATA_TYPE)
        x = x.astype(self.DATA_TYPE)
        self.max_iter = max_iter
        self.n = x.shape[0]
        self.d = x.shape[-1]
        #print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'start intializing')
        if self.init_param == "random":
            #add 1 for uniform
            self.lambda_phi = tf.Variable(np.random.dirichlet(np.array([1.0] * (self.k+1)), self.n), trainable=False,
                                          dtype=self.DATA_TYPE, name="lambda_phi")
            lambda_m_var = np.random.uniform(np.min(x[:, 0]), np.max(x[:, 0]), (self.k, self.d))
        elif self.init_param == "gmm":
            lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k)
            lambda_m_var = c
            self.lambda_phi = tf.Variable(0.01 / self.k * tf.ones((self.n, self.k + 1), dtype=self.DATA_TYPE),
                                          trainable=False, dtype=self.DATA_TYPE, name="lambda_phi")
            self.lambda_phi[:, 1:].assign(lambda_phi_var*0.99)
        elif self.init_param == "self_setting":
            lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k, weights_init=self.weights_init, \
                                                         means_init=self.means_init, \
                                                         precisions_init=self.precisions_init)
            lambda_m_var = c
            self.lambda_phi = tf.Variable(0.01 / self.k * tf.ones((self.n, self.k + 1), dtype=self.DATA_TYPE),
                                          trainable=False, dtype=self.DATA_TYPE, name="lambda_phi")
            self.lambda_phi[:, 1:].assign(lambda_phi_var * 0.99)
        else:
            lambda_phi_var, c = self.init_KMeans_lambda_phi(x, self.k)
            lambda_m_var = c * (np.max(x, axis=0) - np.min(x, axis=0)) + np.min(x, axis=0)
            # add 1 for uniform
            self.lambda_phi = tf.Variable(0.01 / self.k * tf.ones((self.n, self.k+1), dtype=self.DATA_TYPE),
                                          trainable=False, dtype=self.DATA_TYPE, name="lambda_phi")
            for i, label in enumerate(lambda_phi_var):
                self.lambda_phi[i, int(label.numpy()+1)].assign(0.99)

        self.lambda_m = self.add_weight(name="lambda_m",
                                        shape=lambda_m_var.shape,
                                        dtype=self.DATA_TYPE,
                                        initializer=tf.initializers.Constant(lambda_m_var),
                                        trainable=True)
        self._init_param(x)
        #print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'end intializing')

        lbs = []
        n_iters = 0
        for _ in range(self.max_iter):

            Nk = tf.reduce_sum(self.lambda_phi, axis=0) #(k,)
            xbar = tf.matmul(tf.linalg.diag(1/Nk), tf.matmul(tf.transpose(self.lambda_phi), x_tf)) #(k,d)
            xbar_shape = xbar.shape # [3,32]
            Sk = []
            for i in range(self.k+1):
                x_xbar = x_tf-xbar[i] #(n,d)   [200,32]
                rn = self.lambda_phi[:, i] #(n,)
                snk = tf.matmul(tf.expand_dims(x_xbar, axis=-1), tf.expand_dims(x_xbar, axis=-2)) #(n,d,d)
                # snk=[200,32,32]
                Sk.append(tf.reduce_sum(tf.tile(tf.reshape(rn, [-1, 1, 1]), [1, self.d, self.d]) * snk, axis=0)/Nk[i])
            Sk = tf.convert_to_tensor(Sk, dtype=self.DATA_TYPE)#Sk=[14,32,32]
            self.update_lambda_pi(self.lambda_pi, Nk)
            self.update_lambda_u_b_a(self.lambda_u_b_a, Nk, Sk)
            self.update_lambda_beta(self.lambda_beta, Nk)
            self.update_lambda_nu(self.lambda_nu, Nk)
            self.update_lambda_m(self.lambda_m, Nk, xbar)
            self.update_lambda_w(self.lambda_w, Nk, Sk, xbar)
            self.update_lambda_phi2(self.lambda_phi, x, Sk)
            lb = self.elbo(x)
            lbs.append(lb)

            improve = lb - lbs[-2] if n_iters > 0 else lb

            #print("Nk:", Nk/self.n)
            #if n_iters > 0 and 0 <= improve < self.threshold: break
            if n_iters > 0 and -self.threshold <= improve < self.threshold: break
            if n_iters > 0 and improve < -100: break
            #if n_iters > 0 and -np.abs(np.max(lbs)/1000) <= improve < np.abs(np.max(lbs)/1000): break
            
            n_iters += 1
            if self.gif_display:
                title = 'iteration %d' % n_iters
                filename = 'plots/cavi_img%04d.png' % n_iters
                plot_GMM(x, self.lambda_m[:, :2], Sk[1:, :2, :2], \
                         (self.lambda_pi / tf.reduce_sum(self.lambda_pi))[1:], self.true_mean[:, :2], \
                         self.true_var[:, :2, :2], self.k, title = title, savefigpath=filename)



        zn = np.array([np.argmax(self.lambda_phi[q, :]) for q in range(self.n)])
        #print(Counter(zn))
        #plt.style.use('seaborn-darkgrid')
        #plt.figure(figsize=(20, 20))
        #ax=plt.subplot(2, 1, 1)
        #ax.plot([i for i in range(len(lbs))], lbs, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
        #ax.set_title('ELBO')
        #plt.savefig('cavi_plot.png')
        return zn, self.lambda_phi

    def parameters(self):
        '''
        return the trained hyperparameters
        :return: lambda_pi, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o
        '''
        return self.lambda_pi, self.lambda_beta, self.lambda_m, self.lambda_w, self.lambda_nu, self.b_o, self.a_o

    def parameters_o(self):
        '''
        return the original initialized hyperparameters
        :return: alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o
        '''
        return self.alpha_o, self.beta_o, self.m_o, self.w_o, self.nu_o, self.b_o, self.a_o

    def soft_assignment(self):
        '''
        the soft assignment of data
        :return: lambda_phi
        '''
        return self.lambda_phi

    def hard_assignment(self):
        '''
        the hard assignment of data
        :return: argmax(lambda_phi)
        '''
        return np.array([np.argmax(self.lambda_phi[q, :]) for q in range(self.n)])

    def predict(self, x_new, soft_assignment_output = True, likelihood_output = False):
        '''
        return the predicted soft assignment, likelihood or hard assignment of new data.
        Should pay attention that, if the new data is included in the original data, it's better to use
        hard/soft_assignment function because the predict function is based on posterior likelihood whose results
        are a little different from the results of VI.
        :param x_new: new data
        :param soft_assignment_output: return soft assignment or hard assignment
        :param likelihood_output: return the likelihood or not
        :return:soft assignment or hard assignment
        '''
        mu = self.lambda_m
        Lambda = tf.reshape((self.lambda_nu + 1 - self.d) * self.lambda_beta / (1 + self.lambda_beta), [-1, 1, 1]) \
                            * self.lambda_w
        nu = self.lambda_nu + 1 - self.d

        x_new = x_new.astype(self.DATA_TYPE)
        if len(x_new.shape) == 1:
            x_new = tf.reshape(x_new, [1, self.d])
        likelihood = []
        for i in range(x_new.shape[0]):
            gaussian_part = multi_t_density(x_new[i, :], mu, Lambda, nu)
            uniform_part = tf.reshape(1/(tf.math.cumprod(self.b_o-self.a_o)[-1]), [-1])
            likelihood_u_g = tf.concat([uniform_part, gaussian_part], axis=0)
            weights = self.lambda_pi/tf.reduce_sum(self.lambda_pi)
            likelihood.append(weights*likelihood_u_g)
        likelihood = tf.convert_to_tensor(likelihood, dtype=self.DATA_TYPE)

        if likelihood_output:
            return tf.reduce_sum(likelihood, axis=1)
        if soft_assignment_output:
            return likelihood/tf.reduce_sum(likelihood)
        else:
            soft_assignment = likelihood / tf.reduce_sum(likelihood)
            return np.array([np.argmax(soft_assignment[q, :]) for q in range(x_new.shape[0])])


    def init_gmm_lambda_phi(self, x, n_centers, weights_init=None, means_init=None, precisions_init=None):
        gmm = GaussianMixture(n_components=n_centers, covariance_type='full', weights_init=weights_init, \
                              means_init=means_init, precisions_init=precisions_init, reg_covar=self.reg_covar)
        gmm.fit(x)
        return tf.cast(gmm.predict_proba(x), dtype=self.DATA_TYPE), tf.cast(gmm.means_, dtype=self.DATA_TYPE)


    def init_KMeans_lambda_phi(self, x, n_centers, init_times=100, min_delta=1e-3, init_center = None):
        """
        Find an initial value for the lambda phi. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which using the best centerpoint lambda phi is evaluated.
        args:
            x:            tf.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        return:
            lambda_phi:   tf.Tensor (n,)
        """
        if len(x.shape) == 3:
            x = tf.squeeze(x, 1)
        x_min, x_max = tf.reduce_min(x), tf.reduce_max(x)
        x = (x - x_min) / (x_max - x_min)

        random_state = check_random_state(self.random_state)

        min_cost = np.inf
        init_times = np.max([init_times, 10*self.k])

        if init_center is None:
            for i in range(init_times):
                tmp_center = x.numpy()[random_state.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
                l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - tmp_center, ord=2, axis=2)
                l2_cls = tf.argmin(l2_dis, axis=1)

                cost = 0
                for c in range(n_centers):
                    cost += tf.reduce_mean(tf.norm(x[l2_cls == c] - tmp_center[c], ord=2, axis=1))

                if cost < min_cost:
                    min_cost = cost
                    center = tmp_center
        else:
            for i in range(init_times):
                tmp_center = (init_center-x_min)/ (x_max - x_min)
                l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - tmp_center, ord=2, axis=2)
                l2_cls = tf.argmin(l2_dis, axis=1)
                cost = 0
                for c in range(n_centers):
                    cost += tf.reduce_mean(tf.norm(x[l2_cls == c] - tmp_center[c], ord=2, axis=1))
                min_cost = cost
                center = tmp_center

        delta = np.inf
        while delta > min_delta:
            l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - center, ord=2, axis=2)
            l2_cls = tf.argmin(l2_dis, axis=1)
            center_old = tf.convert_to_tensor(center, dtype=tf.double)

            for c in range(n_centers):
                center[c] = tf.reduce_mean(x[l2_cls == c], axis=0)

            delta = tf.reduce_max(tf.reduce_sum(tf.square(center_old - center), axis=1))

        l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - center, ord=2, axis=2)
        l2_cls = tf.argmin(l2_dis, axis=1)
        return tf.cast(l2_cls, dtype=self.DATA_TYPE), tf.cast(center, dtype=self.DATA_TYPE)


    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (k, d, d)
        """
        log_det = []

        for k in range(self.k
        ):

            evals, evecs = tf.linalg.eig(var[k])

            log_det.append(tf.reduce_sum(tf.math.log(tf.math.real(evals))))
        log_det = tf.convert_to_tensor(log_det)
        return tf.expand_dims(log_det, -1)


    def bic(self, xn):
        """
        Bayesian information criterion for a batch of samples. However, the loglikelihood part
        is estimated by the lambda phi, the posterior probabilities assigning to one cluster.
        args:
            x:      tf.Tensor (n, d) 
        returns:
            bic:    float
        
        """
        N = xn.shape[0]
        D = xn.shape[-1]
        xn = tf.cast(xn, dtype=self.DATA_TYPE)

        # Free parameters for covariance, means and mixture components
        free_params = D * self.k + self.k * (
                    D * (D + 1) / 2) + self.k

        # uniform part
        new_lambda_phi_n0 = -tf.math.log(self.lambda_u_b_a) + \
                            dirichlet_expectation_k(self.lambda_pi, 0)
        new_lambda_phi_n0 = tf.tile(tf.reshape(new_lambda_phi_n0, [-1,1]), [N,1])
        #gaussian part;try different D in psi calculation
        new_lambda_phi_nk = tf.reshape(dirichlet_expectation(self.lambda_pi)[1: ], [1, -1])\
        + tf.squeeze(tf.expand_dims(self.lambda_m, axis=-2) @ \
               (tf.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w @ tf.reshape(xn, [-1, 1, D, 1]))) \
        - tf.squeeze(tf.linalg.trace(((1 / 2) * tf.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w) @ \
               tf.matmul(tf.reshape(xn, [-1, 1, D, 1]), tf.reshape(xn, [-1, 1, 1, D])))) \
        - tf.reshape((D / 2) * (1 / self.lambda_beta), [1, -1]) \
        - tf.reshape((1 / 2) * tf.expand_dims(self.lambda_m, axis=-2) @ \
            (tf.reshape(self.lambda_nu, [-1, 1, 1]) * \
             self.lambda_w @ tf.expand_dims(self.lambda_m, axis=-1)), [1, -1]) \
        + (D / 2) * np.log(2.0) \
        + tf.reshape((1 / 2) * tf.math.reduce_sum( \
            [tf.math.digamma(self.lambda_nu / 2 + (1 - i) / 2) for
             i in range(D)], axis=0), [1, -1]) \
        + tf.reshape((1 / 2) * tf.linalg.logdet(self.lambda_w), [1, -1]) \
        - (D / 2) * np.log(2.0 * np.pi)
        # print(new_lambda_phi_n0.shape,new_lambda_phi_nk.shape)
        new_lambda_phi_n0k = tf.concat([new_lambda_phi_n0,new_lambda_phi_nk],axis=1)

        score_temp = tf.reduce_logsumexp(new_lambda_phi_n0k, 1)
        score_temp = tf.math.reduce_mean(score_temp)
        score_temp = tf.cast(score_temp, dtype=self.DATA_TYPE)
        free_params = tf.cast(free_params, dtype=self.DATA_TYPE)

        bic = -2. * score_temp * N + free_params * np.log(N)

        return bic