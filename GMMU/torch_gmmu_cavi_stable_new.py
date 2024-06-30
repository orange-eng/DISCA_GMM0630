import numpy as np
from numpy import random
import numbers
from math import pi
import matplotlib.pyplot as plt
import time
from collections import Counter
from GMMU.util import *
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




class TORCH_CAVI_GMMU(nn.Module):
    def __init__(self, n_cluster, a_o = None, b_o = None, u_filter = False, u_filter_rate=0.0025, \
                 threshold=1e+1, init_param="random", random_state=None, gif_display = False, \
                 true_mean=None, true_var=None, reg_covar = 1e-6, DATA_TYPE = torch.double, weights_init=None, \
                 means_init=None, precisions_init=None):
        """
        Parameter Inizatiloasion
        """
        super(TORCH_CAVI_GMMU, self).__init__()
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




    def init_gmm_lambda_phi(self, x, n_centers, weights_init=None, means_init=None, precisions_init=None):
        gmm = GaussianMixture(n_components=n_centers, covariance_type='full', weights_init=weights_init, \
                              means_init=means_init, precisions_init=precisions_init, reg_covar=self.reg_covar)
        gmm.fit(x)
        return torch.tensor(gmm.predict_proba(x), dtype=self.DATA_TYPE), torch.tensor(gmm.means_, dtype=self.DATA_TYPE)

    def hard_assignment(self):
        '''
        the hard assignment of data
        :return: argmax(lambda_phi)
        '''
        return np.array([np.argmax(self.lambda_phi[q, :]) for q in range(self.n)])





    def _init_param(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x = x.cpu().detach().numpy()
        self.u_chi2_quantile = torch.tensor(49.48)
        #Priors for Uniform
        if self.u_filter:       # u_filter:True
            if self.a_o is None:
                self.a_o = torch.tensor(np.min(x, axis=0), dtype=self.DATA_TYPE)
                # self.a_o = tf.convert_to_tensor(self.a_o, dtype=self.DATA_TYPE)

            if self.b_o is None:
                self.b_o = torch.tensor(np.max(x, axis=0), dtype=self.DATA_TYPE)
                # self.b_o = tf.convert_to_tensor(self.b_o, dtype=self.DATA_TYPE)

            w = generate_random_positive_matrix(self.d)     # w.shape = array(32,32)
            w = torch.from_numpy(w)
            self.u_b_a_o = 1/ (torch.tensor((2.0 * np.pi)**(-self.d/2), dtype=self.DATA_TYPE) * \
                           torch.tensor(torch.det(w)**(-1/2), dtype=self.DATA_TYPE) * \
                           torch.tensor(torch.exp(-0.5*self.u_chi2_quantile), dtype=self.DATA_TYPE))

            # self.lambda_u_b_a = self.add_weight(name="lambda_u_b_a",
            #                                      shape=self.u_b_a_o.shape,
            #                                      dtype=self.DATA_TYPE,
            #                                      trainable=True)
            self.lambda_u_b_a = self.u_b_a_o

        else:
            if self.a_o is None:
                self.a_o = torch.tensor(np.min(x, axis=0), dtype=self.DATA_TYPE)

            if self.b_o is None:
                self.b_o = torch.tensor(np.max(x, axis=0), dtype=self.DATA_TYPE)

            self.u_b_a_o = torch.cumprod(self.b_o - self.a_o)[-1]
            # self.u_b_a_o = tf.math.cumprod(self.b_o - self.a_o)[-1]

            # self.lambda_u_b_a = self.add_weight(name="lambda_u_b_a",
            #                                      shape=self.u_b_a_o.shape,
            #                                      dtype=self.DATA_TYPE,
            #                                      trainable=True)

            self.lambda_u_b_a = self.u_b_a_o

        # Priors
        alpha_o = np.array([1.0] * (self.k+1)) #add 1 for uniform
        nu_o = np.array([float(self.d + 100)])
        w_o = generate_random_positive_matrix(self.d)
        m_o = np.zeros(self.d)#np.mean(x, axis=0)
        beta_o = np.array([0.7])

        # Variational parameters intialization
        lambda_pi_var = np.ones(shape=self.k + 1) #add 1 for uniform
        lambda_beta_var = np.ones(shape=self.k)*0.7
        lambda_nu_var = np.ones(shape=self.k)*2. + self.d
        if self.init_param in ["kmeans", "gmm", "self_setting"]:
            mat = []
            for i in range(self.k):
                xk = x[torch.argmax(self.lambda_phi, axis=1).cpu().detach().numpy() == (i+1), :]
                
                _val1 = (xk - self.lambda_m[i].cpu().detach().numpy()).T
                _val2 = (xk - self.lambda_m[i].cpu().detach().numpy()) 
                _val3 = (xk.shape[0] - 1) + self.reg_covar * torch.eye(xk.shape[1], dtype=self.DATA_TYPE)
                mat.append(
                    np.linalg.inv(_val1 @ _val2 \
                                  / _val3) \
                    / lambda_nu_var[i])
            lambda_w_var = np.array(mat)
        else:
            mat = []
            for i in range(self.k): # k=13
                mat.append(generate_random_positive_matrix(self.d))
            lambda_w_var = np.array(mat)


        self.lambda_pi = torch.tensor(lambda_pi_var).to(device)

        self.lambda_beta = torch.tensor(lambda_beta_var).to(device)
        
        self.lambda_nu = torch.tensor(lambda_nu_var).to(device)

        self.lambda_w = torch.tensor(lambda_w_var).to(device)

        self.alpha_o = torch.tensor(alpha_o, dtype=self.DATA_TYPE).to(device)
        self.nu_o = torch.tensor(nu_o, dtype=self.DATA_TYPE).to(device)
        self.w_o = torch.tensor(w_o, dtype=self.DATA_TYPE).to(device)
        self.m_o = torch.tensor(m_o, dtype=self.DATA_TYPE).to(device)
        self.beta_o = torch.tensor(beta_o, dtype=self.DATA_TYPE).to(device)


    def update_lambda_pi(self, lambda_pi, Nk):
        lambda_pi = self.alpha_o + Nk       # N k in cuda, 
        return lambda_pi


    def update_lambda_u_b_a(self, lambda_u_b_a, Nk, Sk):
        #if np.any(np.isnan(self.lambda_phi)):
        #    print('u_b_a_o and Nk[0]:',self.u_b_a_o,Nk[0])
        if self.u_filter:
            Sk_g = Sk[1:]
            v = torch.max(torch.det(Sk_g))
            self.u_b_a_o = 1 / (torch.tensor((2.0 * np.pi) ** (-self.d / 2), dtype=self.DATA_TYPE) * \
                                torch.tensor(v ** (-1 / 2), dtype=self.DATA_TYPE) * \
                                torch.tensor(np.exp(-0.5 * self.u_chi2_quantile), dtype=self.DATA_TYPE))
        if self.u_b_a_o - Nk[0] > 0:
            lambda_u_b_a=self.u_b_a_o - Nk[0]
        else:
            lambda_u_b_a=self.u_b_a_o
        return lambda_u_b_a

    def update_lambda_beta(self, lambda_beta, Nk):
        #only use the gaussian part
        Nk_g = Nk[1:]
        lambda_beta=self.beta_o + Nk_g

        return lambda_beta

    def update_lambda_nu(self, lambda_nu, Nk):
        # only use the gaussian part
        Nk_g = Nk[1:]
        lambda_nu=self.nu_o + Nk_g

        return lambda_nu

    def update_lambda_m(self, lambda_m, Nk, xbar):
        # only use the gaussian part
        Nk_g = Nk[1:]
        xbar_g = xbar[1:]
        new_m = (self.beta_o*self.m_o + torch.tile(torch.reshape(Nk_g, [-1, 1]), [1, self.d])*xbar_g)/ \
                torch.tile(torch.reshape(self.lambda_beta, [-1, 1]), [1, self.d])
        lambda_m=new_m
        return lambda_m


    def update_lambda_w(self, lambda_w, Nk, Sk, xbar):
        # only use the gaussian part
        Nk_g = Nk[1:]
        xbar_g = xbar[1:]
        Sk_g = Sk[1:]
        K = self.k 
        # inv_w_o = torch.linalg.inv(self.w_o)
        inv_w_o = torch.inverse(self.w_o)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k in range(K):
            NkSk = Nk_g[k]*Sk_g[k]
            e1 = self.beta_o*Nk_g[k]/(self.beta_o+Nk_g[k])
            e2 = torch.matmul(torch.unsqueeze(xbar_g[k]-self.m_o, axis=-1), torch.unsqueeze(xbar_g[k]-self.m_o, axis=-2))
            _val = inv_w_o + NkSk + e1*e2 
            _val2 = torch.eye(inv_w_o.shape[1],device=device)
            _val = _val + self.reg_covar * _val2
            # lambda_w[k, :, :] = torch.linalg.inv(_val)
            lambda_w[k, :, :] = torch.inverse(_val)


        return lambda_w



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

        new_lambda_phi_n0 = -torch.log(self.lambda_u_b_a) + dirichlet_expectation_k_torch(self.lambda_pi, 0)    # new_lambda_phi_n0.shape=[14]
        # uniform part
        # lambda_phi[:, 0] = torch.tile(torch.reshape(new_lambda_phi_n0, (1,new_lambda_phi_n0.shape[0])), (N,))

        lambda_phi[0, :]=new_lambda_phi_n0
        #gaussian part;try different D in psi calculation
        # new_lambda_phi_nk = torch.reshape(dirichlet_expectation_torch(self.lambda_pi)[1: ], [1, -1])\
        # + torch.squeeze(torch.unsqueeze(self.lambda_m, axis=-2) @ \
        # (torch.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w @ torch.reshape(torch.tensor(xn).double(), [-1, 1, D, 1]))) \
        # - torch.squeeze(torch.trace(((1 / 2) * torch.reshape(self.lambda_nu, [-1, 1, 1]) * self.lambda_w) @ \
        # torch.matmul(torch.reshape(torch.tensor(xn).double(), [-1, 1, D, 1]), torch.reshape(torch.tensor(xn).double(), [-1, 1, 1, D])))) \
        # - torch.reshape((D / 2) * (1 / self.lambda_beta), [1, -1]) \
        # - torch.reshape((1 / 2) * torch.unsqueeze(self.lambda_m, axis=-2) @ \
        #     (torch.reshape(self.lambda_nu, [-1, 1, 1]) * \
        #      self.lambda_w @ torch.unsqueeze(self.lambda_m, axis=-1)), [1, -1]) \
        # + (D / 2) * np.log(2.0) \
        # + torch.reshape((1 / 2) * torch.sum( \
        #     [torch.digamma(self.lambda_nu / 2 + (1 - i) / 2) for
        #      i in range(D)], axis=0), [1, -1]) \
        # + torch.reshape((1 / 2) * torch.logdet(self.lambda_w), [1, -1]) \
        # - (D / 2) * np.log(2.0 * np.pi)


        # # tf.math.reduce_sum计算张量tensor沿着某一维度的和，可以在求和后降维。
        # lambda_phi[:, 1:]=torch.reshape(new_lambda_phi_nk,[N,-1])

        # lambda_phi = softmax_torch(lambda_phi)

        lambda_phi = torch.softmax(lambda_phi,dim=-1)

        return lambda_phi

    def update_lambda_phi2(self, lambda_phi, xn, Sk):
        """
        """
        Sk_g = Sk[1:]
        var = Sk_g 
        precision = torch.linalg.inv(var + self.reg_covar * torch.eye(var.shape[1]))
        N = xn.shape[0]
        D = xn.shape[-1]

        new_lambda_phi_n0 = -torch.log(self.lambda_u_b_a) + \
                            dirichlet_expectation_k_torch(self.lambda_pi, 0)
        # uniform part, lambda_phi.shape=(23,14),
        # new_lambda_phi_n0.shape=[14]

        # lambda_phi[:, 0]=torch.tile(torch.reshape(new_lambda_phi_n0, (1, new_lambda_phi_n0.shape[0])), (N, 1))

        lambda_phi[0, :]=new_lambda_phi_n0

        #gaussian part
        # precision.shape=(13,32,32)
        _part1 = torch.reshape(dirichlet_expectation_torch(self.lambda_pi)[1: ], [1, -1])
        _part2 = torch.squeeze(torch.unsqueeze(self.lambda_m, axis=-2) @ (precision @ torch.reshape(torch.tensor(xn).double(), [-1, 1, D, 1])))

        _mat = torch.matmul(torch.reshape(torch.tensor(xn).double(), [-1, 1, D, 1]), torch.reshape(torch.tensor(xn).double(), [-1, 1, 1, D]))
        # _mat.shape=(47,1,32,32)
        _mat2 = ((1 / 2) * precision) @ _mat    # _mat2.shape=(23,13,32,32)
        _part3 = torch.squeeze(torch.trace(_mat2))

        _part4 = torch.reshape((1 / 2) * torch.unsqueeze(self.lambda_m, axis=-2) @ (precision @ torch.unsqueeze(self.lambda_m, axis=-1)), [1, -1])

        _part5 = torch.reshape((1 / 2) * torch.logdet(precision), [1, -1])


        new_lambda_phi_nk = _part1\
        + _part2 \
        - _part3 \
        - _part4 \
        + _part5 - (D / 2) * torch.log(2.0 * np.pi)
        lambda_phi[:, 1:]=torch.reshape(new_lambda_phi_nk,[N,-1])

        # lambda_phi=softmax(lambda_phi)

        return lambda_phi

# /home/lab2/zyc/A_orange/DISCA_GMM
    def elbo(self, xn):
        """
        Evidence Lower Bound definition
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("in")
        D = xn.shape[-1]

        # self.alpha_o=[14]
        # self.lambda_pi=[14]
        # self.lambda_phi=[15,14]
        # self.lambda_pi=[14]
        # self.lambda_w=[13,32,32]
        # self.lambda_m=[13,32]


        e3 = torch.tensor(0., dtype=self.DATA_TYPE).to(device)
        e2 = torch.tensor(0., dtype=self.DATA_TYPE).to(device)
        h2 = torch.tensor(0., dtype=self.DATA_TYPE).to(device)
        _alpha = self.alpha_o-1
        _lambda = dirichlet_expectation_torch(self.lambda_pi)
        e1 = -log_beta_function_torch(self.alpha_o) + torch.squeeze(_alpha*_lambda)
        h1 = log_beta_function_torch(self.lambda_pi) - torch.squeeze((self.lambda_pi-1)*dirichlet_expectation_torch(self.lambda_pi))

        logdet = torch.tensor([torch.logdet(self.lambda_w[i, :, :]) for i in range(self.k)], dtype=self.DATA_TYPE).to(device)

        logDeltak = torch.tensor(D * torch.log(torch.tensor(2.0)), dtype=self.DATA_TYPE).to(device) + logdet
        for i in range(1, D + 1):
            logDeltak = logDeltak + torch.digamma((self.lambda_nu + 1 - i) / 2.).to(device)
        logDeltak = torch.tensor(logDeltak, dtype=self.DATA_TYPE).to(device)


        _e2 = torch.reshape(dirichlet_expectation_torch(self.lambda_pi), [-1, 1])# _e2=[14,1]
        e2 = e2 + torch.squeeze(self.lambda_phi @ _e2)
        h2 = h2 - torch.trace(self.lambda_phi @ torch.log(torch.transpose(self.lambda_phi,0,1)))
        product = torch.squeeze(torch.unsqueeze((torch.unsqueeze(torch.tensor(xn), axis=1) - self.lambda_m), axis=-2) @ self.lambda_w @ \
        torch.unsqueeze((torch.unsqueeze(torch.tensor(xn), axis=1) - self.lambda_m), axis=-1))
        aux = logDeltak - torch.tensor(D * torch.log(torch.tensor(2. * np.pi)), dtype=self.DATA_TYPE) \
              - self.lambda_nu * product - D / self.lambda_beta
        e3 = e3 + torch.sum(torch.tensor(1 / 2, dtype=self.DATA_TYPE) * self.lambda_phi[:, 1:] * aux)
        #uniform part may influence the elbo dramatically
        e3 = e3 - torch.sum(self.lambda_phi[:, 0])*torch.log(self.lambda_u_b_a)
        product = torch.tensor([torch.matmul(torch.unsqueeze(self.lambda_m[K,:]-self.m_o, axis=0), torch.matmul(self.lambda_w[K,:,:],
                    torch.unsqueeze(self.lambda_m[K,:]-self.m_o, axis=-1))) for K in range(self.k)], dtype=self.DATA_TYPE).to(device)

        # print("8")
        traces = torch.tensor([torch.trace(torch.matmul(torch.linalg.inv(self.w_o+self.reg_covar * torch.eye(self.w_o.shape[1], dtype=self.DATA_TYPE).to(device)),self.lambda_w[K,:,:])) for K in range(self.k)]).to(device)
        # print("7")
        h4 = -torch.squeeze(0.5*logDeltak + D/2* torch.log(self.lambda_beta/(2*np.pi))-torch.tensor(D/2., dtype=self.DATA_TYPE))
        ###-lnB
        logB = self.lambda_nu / 2 * logdet + D * self.lambda_nu / 2 * torch.log(torch.tensor(2., dtype=self.DATA_TYPE)) + D * (
                    D - 1) / 4 * torch.log(
            torch.tensor(np.pi, dtype=self.DATA_TYPE))
        for i in range(1, D + 1):
            logB = logB + torch.lgamma((self.lambda_nu + 1 - i) / 2.)
        logB = torch.tensor(logB, dtype=self.DATA_TYPE)
        ###part2: -lnB-(v-d-1)/2*E[ln|lambda_k|]+vD/2
        h5 = torch.sum(logB - (self.lambda_nu - D - 1) / 2 * logDeltak + self.lambda_nu * D / 2)
        h5 = torch.tensor(h5, dtype=self.DATA_TYPE)

        ##E[ln p(mu,lambda)]
        ###part1
        _val1 = D*torch.log(torch.tensor(self.beta_o).to(device)).to(device) 
        _val2 = torch.tensor(D * torch.log(torch.tensor(2 * np.pi).to(device)), dtype=self.DATA_TYPE).to(device) 
        _val3 = self.beta_o * self.lambda_nu * product
        _val4 = D * self.beta_o / self.lambda_beta
        e4 = torch.squeeze(0.5 * (_val1+ logDeltak - _val2- _val3 - _val4))
        e4 = torch.tensor(e4, dtype=self.DATA_TYPE).to(device)
        ###-ln B_0
        logB = self.nu_o / 2 * torch.log(torch.det(self.w_o)) + \
               D * self.nu_o / 2 * torch.tensor(torch.log(torch.tensor(2.)), dtype=self.DATA_TYPE) + D * (
                    D - 1) / 4 * torch.log(torch.tensor((np.pi), dtype=self.DATA_TYPE))
        for i in range(1, D + 1):
            logB = logB + torch.lgamma((self.nu_o + 1 - i) / 2.)
        logB = torch.tensor(logB, dtype=self.DATA_TYPE)
        ###part2 K*ln B_0 + ...
        e5 = torch.squeeze(-logB + (self.nu_o - D - 1) / 2 * logDeltak - self.lambda_nu / 2 * traces)
        e5 = torch.tensor(e5, dtype=self.DATA_TYPE)
        # print("1")
        # LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5
        LB = e1 + h1

        # e1=[14]
        # e2=[46]
        # e3=[]
        # e4=[13]
        # e5=[13]
        # h1=[14]
        # h2=[]
        # h4=[13]
        # h5=[]

        #print(e1, e2, e3, e4, e5, h1, h2, h4, h5)
        return LB



    def fit(self, x, max_iter=10):  # x=[4,32]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x_tf = torch.tensor(x, dtype=self.DATA_TYPE).to(device)
        # x =  x.numpy()
        self.max_iter = max_iter
        self.n = x.shape[0]     # n=64
        self.d = x.shape[-1]    # d=32


        if self.init_param == "random":
            _random = np.array([1.0] * (self.k+1))  # shape=(14,), k=13
            _random2 = np.random.dirichlet(_random, self.n)
            self.lambda_phi = torch.tensor(_random2).to(device)
            _min = torch.min(x[:, 0])
            _max = torch.max(x[:, 0])

            lambda_m_var = np.random.uniform(_min.cpu().detach().numpy(), _max.cpu().detach().numpy(), size=(self.k, self.d))
        elif self.init_param == "gmm":
            lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k)
            lambda_m_var = c
            self.lambda_phi = torch.tensor(0.01 / self.k * torch.ones((self.n, self.k + 1), dtype=self.DATA_TYPE), dtype=self.DATA_TYPE).to(device)
            self.lambda_phi[:, 1:]=lambda_phi_var*0.99

        elif self.init_param == "self_setting":
            # lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k, weights_init=self.weights_init, \
            #                                              means_init=self.means_init, \
            #                                              precisions_init=self.precisions_init)
            lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k)
            lambda_m_var = c
            self.lambda_phi =torch.tensor(0.01 / self.k * torch.ones((self.n, self.k + 1), dtype=self.DATA_TYPE), dtype=self.DATA_TYPE).to(device)
            self.lambda_phi[:, 1:]=lambda_phi_var*0.99
        else:
            lambda_phi_var, c = self.init_gmm_lambda_phi(x, self.k)
            lambda_m_var = c * (np.max(x, axis=0) - np.min(x, axis=0)) + np.min(x, axis=0)
            # add 1 for uniform
            self.lambda_phi =torch.tensor(0.01 / self.k * torch.ones((self.n, self.k + 1), dtype=self.DATA_TYPE),dtype=self.DATA_TYPE).to(device)
            for i, label in enumerate(lambda_phi_var):
                self.lambda_phi[i, int(label.numpy()+1)] = (0.99)


        # self.lambda_m = self.add_weight(name="lambda_m",
        #                                 shape=lambda_m_var.shape,
        #                                 dtype=self.DATA_TYPE,
        #                                 trainable=True)

        self.lambda_m = lambda_m_var        # lambda_m_var = (13,32)
        # torch.nn.Parameter(ones)


        self._init_param(x)

        # self.lambda_phi =(64,14), x_tf=(64,32)
        lbs = []
        n_iters = 0
        for _ in range(self.max_iter):
            Nk = torch.sum(self.lambda_phi, axis=0).to(device) #(k,)
            xbar = torch.matmul(torch.diag(1/Nk), torch.matmul(torch.transpose(self.lambda_phi,0,1), x_tf)) #(k,d)
            # xbar = [3,32]
            Sk = []
            for i in range(self.k+1):
                x_xbar = x_tf-xbar[i] #(n,d)        (4,32)
                rn = self.lambda_phi[:, i] #(n,)


                _bar1 = torch.unsqueeze(x_xbar, axis=-1)
                _bar2 = torch.unsqueeze(x_xbar, axis=-2)
                snk = torch.mul(_bar1, _bar2) #(n,d,d)
                # Sk.append(torch.sum(torch.tile(torch.reshape(rn, [-1, 1, 1]), [1, self.d, self.d]) * snk, axis=0)/Nk[i])

                _val = torch.sum(torch.tile(torch.reshape(rn, [-1, 1, 1]), [1, self.d, self.d]) * snk, axis=0)/Nk[i]

                _val = torch.unsqueeze(_val, dim=0)
                Sk.append(_val)

            # Sk = torch.tensor(Sk, dtype=self.DATA_TYPE)
            Sk = torch.cat(Sk)  # Sk=[14,32,32]
            self.lambda_pi = self.update_lambda_pi(self.lambda_pi, Nk)
            # self.lambda_pi = self.alpha_o + Nk

            self.lambda_u_b_a = self.update_lambda_u_b_a(self.lambda_u_b_a, Nk, Sk)
            self.lambda_beta = self.update_lambda_beta(self.lambda_beta, Nk)
            self.lambda_nu = self.update_lambda_nu(self.lambda_nu, Nk)
            self.lambda_m = self.update_lambda_m(self.lambda_m, Nk, xbar)
            self.lambda_w = self.update_lambda_w(self.lambda_w, Nk, Sk, xbar)
            # self.update_lambda_phi2(self.lambda_phi, x, Sk)
            self.lambda_phi = self.update_lambda_phi(self.lambda_phi, x)
            lb = self.elbo(x)
            lbs.append(lb)

            improve = lb - lbs[-2] if n_iters > 0 else lb


            if n_iters > 0 and -self.threshold <= improve < self.threshold: break
            if n_iters > 0 and improve < -100: break




        zn = np.array([np.argmax(self.lambda_phi[q, :].cpu().detach().numpy()) for q in range(self.n)])



        return zn, self.lambda_phi      # self.lambda_phi=[19,14]


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
        return np.array([np.argmax(self.lambda_phi[q, :].cpu().detach().numpy()) for q in range(self.n)])


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
        Lambda = torch.reshape((self.lambda_nu + 1 - self.d) * self.lambda_beta / (1 + self.lambda_beta), [-1, 1, 1]) \
                            * self.lambda_w
        nu = self.lambda_nu + 1 - self.d

        x_new = x_new.astype(self.DATA_TYPE)
        if len(x_new.shape) == 1:
            x_new = torch.reshape(x_new, [1, self.d])
        likelihood = []
        for i in range(x_new.shape[0]):
            gaussian_part = multi_t_density(x_new[i, :], mu, Lambda, nu)
            uniform_part = torch.reshape(1/(torch.cumprod(self.b_o-self.a_o)[-1]), [-1])
            likelihood_u_g = torch.concat([uniform_part, gaussian_part], axis=0)
            weights = self.lambda_pi/torch.sum(self.lambda_pi)
            likelihood.append(weights*likelihood_u_g)
        likelihood = torch.tensor(likelihood, dtype=self.DATA_TYPE)

        if likelihood_output:
            return torch.sum(likelihood, axis=1)
        if soft_assignment_output:
            return likelihood/torch.sum(likelihood)
        else:
            soft_assignment = likelihood / torch.sum(likelihood)
            return np.array([np.argmax(soft_assignment[q, :]) for q in range(x_new.shape[0])])




    def init_gmm_lambda_phi(self, x, n_centers, weights_init=None, means_init=None, precisions_init=None):
        gmm = GaussianMixture(n_components=n_centers, covariance_type='full', weights_init=weights_init, \
                              means_init=means_init, precisions_init=precisions_init, reg_covar=self.reg_covar)
        gmm.fit(x.cpu().detach().numpy())   # x=[20,32]
        return torch.tensor(gmm.predict_proba(x.cpu().detach().numpy()), dtype=self.DATA_TYPE), torch.tensor(gmm.means_, dtype=self.DATA_TYPE)







