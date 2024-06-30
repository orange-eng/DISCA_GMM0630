
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

def cluster_mean_cov(X, Y):
    Y_class = list(set(Y))
    mean = []
    cov = []
    for k in Y_class:
        X_k = X[Y==k]
        if len(X_k) <=1:
            continue
        mean.append(np.mean(X_k, axis=0))
        cov.append(np.cov(X_k.T))
    return np.array(mean), np.array(cov)


def draw_ellipse(mu, cov, num_sd=1):
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

cols=['#6A539D','#E6D7B2','#99CCCC','#FFCCCC','#DB7093','#D8BFD8','#6495ED','#1E90FF','#7FFFAA','#FFFF00']
cols=np.array(cols)

def plot_GMM(X, mu, lam, centres = None, covs = None, title = None, cols=cols, savefigpath=False, size=None, Y_pred = None):
    '''
    only plot the first 2 dim
    X: data set (n,p)
    mu: cluster sample mean (k,2)
    lam: cluster sample covariance  (k,2,2)
    centres: true cluster mean (k,2)
    coves: true cluster cov (k,2,2)
    '''
    plt.figure(figsize = size)
    plt.xlim(np.min(X[:, 0])-3, np.max(X[:, 0])+3)
    plt.ylim(np.min(X[:, 1])-3, np.max(X[:, 1])+3)
    if Y_pred is None:
        plt.scatter(X[:, 0], X[:, 1], marker='o', s=1)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=cols[Y_pred], marker='o', s=1)

    K = mu.shape[0]
    for k in range(K):
        cov = lam[k]
        x_ell, y_ell = draw_ellipse(mu[k], cov)
        plt.plot(x_ell, y_ell, cols[k])

    # Plotting the ellipses for the GMM that generated the data
    if (centres is not None) and (covs is not None):
        for i in range(centres.shape[0]):
            x_true_ell, y_true_ell = draw_ellipse(centres[i], covs[i])
            plt.plot(x_true_ell, y_true_ell, 'g--')

    plt.title(title)
    if isinstance(savefigpath, str):
        plt.savefig(savefigpath, transparent=True)
    else:
        plt.show()