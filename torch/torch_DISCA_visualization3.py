# 2024/6/20
#   GMMU: the process always shut down suddenly, and there is no bug information.
# NN Classification is OK

# Chengzhi Cao





import sys
sys.path.append('/home/lab2/zyc/A_orange/DISCA_GMM')

import faulthandler
faulthandler.enable()
import sys,argparse
import pdb
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure
from utils.plots import *
from utils.metrics import *
import h5py
from disca_dataset.DISCA_visualization import *
import pickle, mrcfile
import scipy.ndimage as SN
from PIL import Image
from collections import Counter
# from tf.disca.DISCA_gmmu_cavi_llh_scanning_new import *
# from GMMU.gmmu_cavi_stable_new import CAVI_GMMU as GMM

from GMMU.torch_gmmu_cavi_stable_new import TORCH_CAVI_GMMU as GMM

from config import *
from tqdm import *
import cv2
import warnings


from torch_DISCA_gmmu import *


import numpy as np
import scipy


import sys, multiprocessing, importlib
from multiprocessing.pool import Pool
from skimage.transform import rescale 


from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA 

import scipy.ndimage as SN
import gc


import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage,os


warnings.filterwarnings("ignore")

np.random.seed(42)
color=['#6A539D','#E6D7B2','#99CCCC','#FFCCCC','#DB7093','#D8BFD8','#6495ED',\
'#1E90FF','#7FFFAA','#FFFF00','#FFA07A','#FF1493','#B0C4DE','#00CED1','#FFDAB9','#DA70D6']
color=np.array(color)




def update_beta(beta_0, Nk):
    lambda_beta = beta_0 + Nk
    return lambda_beta

def update_nu(nu_0, Nk):
    lambda_nu = nu_0 + Nk
    return lambda_nu


def update_w_inv(responsibility, w_o, beta_o, m_o, x):

    d = x.shape[-1]
    Nk = torch.sum(responsibility, axis=0) #shape:([])
    xbar = torch.matmul(torch.unsqueeze(responsibility, axis=0), x)/ Nk #(d,)
    x_xbar = x - xbar  # (n,d)
    snk = torch.matmul(torch.unsqueeze(x_xbar, axis=-1), torch.unsqueeze(x_xbar, axis=-2)) # (n,d,d)
    Sk=torch.sum(torch.tile(torch.reshape(responsibility, [-1, 1, 1]), [1, d, d]) * snk, axis=0) / Nk
    #print(responsibility)
    inv_w_o = torch.linalg.inv(w_o)
    NkSk = Nk*Sk
    e1 = beta_o*Nk/(beta_o+Nk)
    e2 = torch.matmul(torch.unsqueeze(xbar-m_o, axis=-1), torch.unsqueeze(xbar-m_o, axis=-2))
    #print(inv_w_o, NkSk, e1*e2)
    lambda_w_inv=inv_w_o + NkSk + e1*e2
    return lambda_w_inv



def multi_log_gamma_function(input, p):
    """
    input: tf.Tensor (1,)
    p:     tf.Tensor (1,)
    """
    C = torch.log(np.pi) * p * (p-1) /4
    C = torch.tensor(C, dtype=input.dtype)
    for i in range(p):
        C = C + torch.lgamma(input-i/2)
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
    lml = - N*D/2*torch.log(np.pi)
    lml = torch.tensor(lml, dtype=x.dtype)
    beta_0 = torch.tensor(beta_0, dtype=x.dtype)
    w_0 = torch.tensor(w_0, dtype=x.dtype)
    nu_0 = torch.tensor(nu_0, dtype=x.dtype)
    lambda_beta = torch.tensor(lambda_beta, dtype=x.dtype)
    lambda_w = torch.tensor(lambda_w, dtype=x.dtype)
    lambda_nu = torch.tensor(lambda_nu, dtype=x.dtype)

    lml = lml + multi_log_gamma_function(lambda_nu/2, D) - multi_log_gamma_function(nu_0/2, D)
    
    lml = lml + nu_0/2*torch.linalg.logdet(torch.linalg.inv(w_0)) - \
        lambda_nu/2*torch.linalg.logdet(torch.linalg.inv(lambda_w))

    lml = lml + (D/2)*(torch.log(beta_0)-torch.log(lambda_beta))
    
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
    lml = - N*D/2*torch.log(np.pi)
    lml = torch.tensor(lml, dtype=x.dtype)
    beta_0 = torch.tensor(beta_0, dtype=x.dtype)
    w_0 = torch.tensor(w_0, dtype=x.dtype)
    nu_0 = torch.tensor(nu_0, dtype=x.dtype)
    lambda_beta = torch.tensor(lambda_beta, dtype=x.dtype)
    lambda_w_inv = torch.tensor(lambda_w_inv, dtype=x.dtype)
    lambda_nu = torch.tensor(lambda_nu, dtype=x.dtype)

    lml = lml + multi_log_gamma_function(lambda_nu/2, D) - multi_log_gamma_function(nu_0/2, D)
    lml = lml + nu_0/2*torch.log(torch.linalg.det(torch.linalg.inv(w_0))) - \
        lambda_nu/2*torch.linalg.logdet(lambda_w_inv)
    lml = lml + (D/2)*(torch.log(beta_0)-torch.log(lambda_beta))
    return lml

def distance_matrix(t, n_components, n_feature):
        t1 = torch.reshape(t, (1, n_components, n_feature))
        t2 = torch.reshape(t, (n_components, 1, n_feature))
        distance_matrix = torch.norm(t1 - t2)
        return distance_matrix



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

    alpha = torch.tensor(alpha, x.dtype)

    prob_c_merge = torch.sum(torch.log(torch.range(1, N, dtype=x.dtype)))
    prob_c1 = torch.sum(torch.log(torch.range(1, n_c1, dtype=x.dtype)))
    prob_c2 = torch.sum(torch.log(torch.range(1, n_c2, dtype=x.dtype)))

    #log likelihood of cluster_merge
    cr = r_all  # (n, )
    cNk = torch.sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    cluster_merge_log_prob = log_marginal_likelihood2(X_merge, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster1
    cr = r_c1  # (n, )
    cNk = torch.sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    c1_log_prob = log_marginal_likelihood2(X_c1, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster2
    cr = r_c2  # (n, )
    cNk = torch.sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    c2_log_prob = log_marginal_likelihood2(X_c2, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    log_acceptpro = -torch.log(alpha)-prob_c1-prob_c2+prob_c_merge \
                    -c1_log_prob-c2_log_prob+cluster_merge_log_prob\
                    - torch.tensor((n_c1+n_c2-2)*torch.log(2.), dtype=x.dtype)
    if np.isnan(log_acceptpro):
        if np.isnan(c1_log_prob):
            print('c1_log_prob',c1_log_prob)
        if np.isnan(c2_log_prob):
            print('c2_log_prob',c2_log_prob)
        if np.isnan(cluster_merge_log_prob):
            print('cluster_merge_log_prob',cluster_merge_log_prob)

    return torch.exp(log_acceptpro)


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

    alpha = torch.tensor(alpha, x.dtype)

    prob_c_original = torch.sum(torch.log(torch.range(1, N, dtype=x.dtype)))
    prob_c_split1 = torch.sum(torch.log(torch.range(1, n_sub1, dtype=x.dtype)))
    prob_c_split2 = torch.sum(torch.log(torch.range(1, n_sub2, dtype=x.dtype)))

    #log likelihood of cluster_original
    cr = r_all  # (n, )
    cNk = torch.sum(cr, axis=0) #(1, )
    c_beta = update_beta(beta_0, cNk) #(1,)
    c_nu = update_nu(nu_0, cNk) #(1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x) #(d, d)

    cluster_split_log_prob = log_marginal_likelihood2(X_split, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster1
    cr = r_c1  # (n, )
    cNk = torch.sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    sub1_log_prob = log_marginal_likelihood2(X_sub1, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)

    # log likelihood of sub-cluster2
    cr = r_c2  # (n,)
    cNk = torch.sum(cr, axis=0)  # (1, )
    c_beta = update_beta(beta_0, cNk)  # (1,)
    c_nu = update_nu(nu_0, cNk)  # (1,)
    c_w_inv = update_w_inv(cr, w_0, beta_0, m_0, x)  # (d, d)

    sub2_log_prob = log_marginal_likelihood2(X_sub2, beta_0, w_0, nu_0, c_beta, c_w_inv, c_nu)
    
    log_acceptpro = torch.log(alpha)+prob_c_split1+prob_c_split2-prob_c_original \
                    + sub1_log_prob+sub2_log_prob-cluster_split_log_prob \
                    + torch.tensor((n_sub1+n_sub2-2)*torch.log(2.), dtype=x.dtype)
    if np.isnan(log_acceptpro):
        if np.isnan(sub1_log_prob):
            print('sub1_log_prob',sub1_log_prob)
        if np.isnan(sub2_log_prob):
            print('sub2_log_prob',sub2_log_prob)
        if np.isnan(cluster_split_log_prob):
            print('cluster_split_log_prob',cluster_split_log_prob)


    return torch.exp(log_acceptpro)

def statistical_fitting_split_merge(features, labels, candidateKs, K, reg_covar, it, u_filter = True,u_filter_rate=0.0025, alpha = 1.0):

    features_pca = features
    n_features = features_pca.shape[1]

    reg_covar = np.max([reg_covar * (0.5**it), 1e-6])    
    labels_K = [] 
    models = []
    BICs = [] 
    k_new = K
    k_original = K

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if K in candidateKs:
        gmm_0 = GMM(n_cluster = K, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate, \
                    reg_covar=reg_covar)
        gmm_0.fit(features_pca) 
        labels_k_0 = gmm_0.hard_assignment()
        alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = gmm_0.parameters_o()
        r_cluster = gmm_0.soft_assignment()
        hardcluster_label = gmm_0.hard_assignment()


        for i in range(1, K+1): #ignore uniform
            x_i = features_pca[hardcluster_label == i,:]
            if x_i.shape[0] <= 1 :
                    continue
            gmm_i = GaussianMixture(n_components=2, covariance_type='full', \
                                    tol=0.001, random_state=i, init_params = 'kmeans', reg_covar=reg_covar)
            gmm_i.fit(x_i.cpu().detach().numpy()) 
            subcluster_i_hard_label = gmm_i.predict(x_i.cpu().detach().numpy())
            x_i1 = x_i[subcluster_i_hard_label==0, :]
            x_i2 = x_i[subcluster_i_hard_label==1, :]
            if x_i1.shape[0] == 0 or x_i2.shape[0] == 0:
                    continue
            r_all = r_cluster[:, i]  # (n, )
            r_sub = torch.unsqueeze(r_all, axis=-1) * torch.tensor(gmm_i.predict_proba(features_pca.cpu().detach().numpy()))  # (n, 2)
            r_c1 = r_sub[:, 0]  # (n, )
            r_c2 = r_sub[:, 1]  # (n, )

            hs = acceptpro_split_hs(features_pca, x_i, x_i1, x_i2, r_all, r_c1, r_c2, \
                                    m_o, beta_o, w_o, nu_o, alpha)
            #if not np.isnan(hs):
            #    print('hs', hs)
            if torch.random(shape=[], minval=0., maxval=1) < torch.tensor(hs, torch.float32):
                if k_new >= np.max(candidateKs):
                    break
                else:
                    k_new = k_new + 1


        if k_new != K:
            gmm_1 = GMM(n_cluster = k_new, a_o = None, b_o = None, u_filter = u_filter, \
                        u_filter_rate = u_filter_rate, reg_covar=reg_covar)
            # gmm_1.to(device)
            gmm_1.fit(features_pca)
        else:
            gmm_1 = gmm_0

        alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = gmm_1.parameters_o()
        lambda_alpha, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o = gmm_1.parameters()
        softcluster_label = gmm_1.soft_assignment() #tensor
        hardcluster_label = gmm_1.hard_assignment() #numpy
        k_new2 = k_new

        #merge
        dm = distance_matrix(lambda_m, k_new, n_features)
        merged_list = []
        nan_cluster_list = [] #the clusters that contain no point
        if k_new > 1:
            n_pair = min(3, k_new - 1)
            merge_pair = torch.argsort(dm)[:, 1: (n_pair + 1)]
            for i, pairs in enumerate(merge_pair):
                if (i + 1) in list(np.array(merged_list).flat) or (i + 1) in nan_cluster_list:
                    # plus 1 because considering uniform pi_0
                    continue
                if k_new2 <= np.min(candidateKs):
                    break
                mask_i = hardcluster_label == (i + 1)  # plus 1 because considering uniform pi_0
                X_c1 = torch.boolean_mask(features_pca, mask=mask_i)
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
                    X_c2 = torch.boolean_mask(features_pca, mask=mask_j)
                    X_merge = torch.concat([X_c1, X_c2], 0)

                    if X_c2.shape[0] == 0 :
                        nan_cluster_list.append(pair + 1)
                        k_new2 = k_new2 - 1
                        continue

                    r = softcluster_label[:, 1:]#(n,k)
                    r_c1 = r[:, i] #(n,)
                    r_c2 = r[:, pair] #(n,)
                    r_merge = r_c1 + r_c2 #(n,)

                    hm = acceptpro_merg_hm(features_pca, X_merge, X_c1, X_c2, \
                                            r_merge, r_c1, r_c2, \
                                            m_o, beta_o, w_o, nu_o, alpha)
                    #if not np.isnan(hm):
                    #    print('hm', str([i+1, pair+1]), hm)
                    if (X_c1.shape[0] == 0) or (X_c2.shape[0] == 0) or \
                            (torch.random(shape=[], minval=0., maxval=1) < torch.tensor(hm, torch.float32)):
                        merged_list.append([i + 1, pair + 1])
                        if k_new2 <= np.min(candidateKs):
                            break
                        else:
                            k_new2 = k_new2 - 1
        
        if k_new2 != k_new:
            gmm = GMM(n_cluster = k_new2, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate, reg_covar=reg_covar)
            # gmm.to(device)
            gmm.fit(features_pca)
        else:
            gmm = gmm_1
        
        labels_k = gmm.hard_assignment()
        K_temp = k_new2 #ignore uniform 


    else: 
        K_temp = int(0.5*(np.min(candidateKs) + np.max(candidateKs)))
        gmm = GMM(n_cluster = K_temp, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate,reg_covar=reg_covar) 
        # gmm = gmm.to(device)
        gmm.fit(features_pca) 
        labels_k = gmm.hard_assignment()

    if K_temp == K and (len(np.unique(labels_k))-1)==K_temp: 
        same_K = True

        new_model_dif_K = True
        re_num = 0
        while(new_model_dif_K and (re_num<=10)):
            if K_temp != (len(np.unique(labels))-1):
                break
            #retrain with the pre-setting parameters
            weights_init = np.array([np.sum(labels == j) / (len(labels)) for j in range(1,K+1)])
            weights_init = weights_init /np.sum(weights_init)
            means_init = np.array([np.mean(features_pca[labels == j].cpu().detach().numpy(), 0) for j in range(1,K+1)])
            precisions_init = np.array(
                [np.linalg.pinv(np.cov(features_pca[labels == j].T.cpu().detach().numpy())) for j in range(1,K+1)])
            gmm = GMM(n_cluster=K, a_o=None, b_o=None, u_filter=u_filter, u_filter_rate=u_filter_rate, \
                    reg_covar=reg_covar, init_param="self_setting",\
                    weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)
            # gmm.to(device)
            gmm.fit(features_pca)
            labels_k = gmm.hard_assignment()
            if K==len(np.unique(labels_k))-1:
                new_model_dif_K = False
            re_num = re_num+1
    else: 
        same_K = False 
        K = K_temp
    labels_temp = remove_empty_cluster(labels_k)
    labels_temp_proba = gmm.soft_assignment().cpu().numpy()
    # if k != the number of Gaussian cluster predicted, it means there is some clusters disappearing, 
    # so ignore this iteration
    # if K != (len(np.unique(labels_temp))-1):
    #     same_K = False
    #     labels_temp = labels

    print('Estimated K:', K)
    #if np.any(np.isnan(labels_temp_proba)):
    #    print('###in###there is nan in labels_temp_proba')
    #    print(labels_temp_proba)
    return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm   




# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




DIVIDE = 40
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_yaml', type=str, default=r'/home/lab2/zyc/A_orange/DISCA_GMM/config/train.yaml', help='YAML config file')
config_parser = parser.parse_args(args = []) #parser.parse_args() in py file
args = jupyter_parse_args_yaml(config_parser)


abel_path = args.saving_path+'/results'
model_path = args.saving_path+'/models'
label_names = ['labels_'+args.algorithm_name]
figures_path = args.saving_path+'/figures/'+label_names[0]
infos = pickle_load(args.data_path+'/info.pickle')
v = read_mrc_numpy_vol(args.data_path+'/emd_4603.map')
algorithms = ['classificationmodel']+args.algorithm_name.split('_')
v = (v - np.mean(v))/np.std(v)
vs = []
s = 32//2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#trained model
model_names = []
for model_name in os.listdir(model_path):
    algo = model_name.split('_')[:len(algorithms)]
    if algo == algorithms :
        model_names.append(os.path.splitext(model_name)[0])
print('model_names=',model_names)

#extracted particles
h5f = h5py.File(args.filtered_data_path,'r')                                                        
# x_train = h5f['dataset_1'][:] 
x_train = h5f['dataset_1'][:16000] 
infos = infos[:16000]
h5f.close()
print('x_train=',len(x_train))

total_num = len(x_train)//DIVIDE


total_label = []


#visualization using NN
for model_name in model_names:
    classmodelpath = os.path.join(model_path,model_name)+'.h5'
    yopopath = os.path.join(model_path,'deltamodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
    #gpath = os.path.join(model_path,'gmmumodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
    figure_path = os.path.join(figures_path,'_'.join(model_name.split('_')[1:]))
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    
    yopo = YOLO_Model()
    classmodel = YOPO_classification(num_labels=20, vector_size = 32)
    # parallel_model = CombinedModel(yopo,classmodel)
    yopo.to(device)
    classmodel.to(device)

    yopo_stat = torch.load(yopopath)
    class_stat = torch.load(classmodelpath)
    yopo.load_state_dict(yopo_stat)
    classmodel.load_state_dict(class_stat)
    # scanning_model = CombinedModel(yopo,classmodel)

    yopo.eval()
    with torch.no_grad():
        for a in range(total_num):
            print("a=",a)
            h5f = h5py.File(args.filtered_data_path,'r')    
            x_train = h5f['dataset_1'][a*DIVIDE : a*DIVIDE+DIVIDE] # only 'dataset_1'                              
            h5f.close()
            # print(x_train.shape)

            _infos = infos[a*DIVIDE : a*DIVIDE+DIVIDE]
            # infos = infos[:]
            infonp = np.array(_infos)
            # print(set(infonp[:,0]),infonp.shape)

            # x_train = torch.tensor(x_train,dtype=torch.float32).to(device)    # [197,24,24,24,1]

            x_train = torch.tensor(x_train,dtype=torch.float32).to(device)    # [197,24,24,24,1]
            x_train = x_train.permute(0,4,1,2,3)  
            features = yopo(x_train)
            labels_soft = classmodel(features)
            labels = np.array([np.argmax(labels_soft[q, :].cpu().detach().numpy()) for q in range(len(labels_soft))])
            # find the index of the max value


            total_label = np.concatenate((total_label,labels))



# labels = total_label.astype(int)
labels = np.array(total_label,dtype=np.int)

# labels = array([0,1,2,3,4,....])
############################################################
############################################################
############################################################
for i in tqdm(range(np.max(labels) + 1)):
    locs = np.array(infos)[labels == i]
    v_i = np.zeros_like(v)      # v_i=[928,928,464]
    for j in locs:
        # print("j=",j)
        if j[0] == 'emd_4603.map': #emd_4603_deconv_corrected.mrc / emd_4603.map
            v_i[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s] = \
            v[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s]
    save_png(cub_img(v_i[:,:,::15])['im'], os.path.join(figure_path, 'NN'+str(i) + model_name + '.png'))
    # save_png(cub_img(v_i[:,:,:])['im'], os.path.join(figure_path, 'GMMU'+str(i) + model_name + '.png'))





'''
x_train = h5f['dataset_1'][:40] # only 'dataset_1'                              
h5f.close()
print(x_train.shape)

infos = infos[:40]
# infos = infos[:]
infonp = np.array(infos)
print(set(infonp[:,0]),infonp.shape)


########################################################
########################################################
########################################################

#visualization using GMMU
for model_name in model_names:
    classmodelpath = os.path.join(model_path,model_name)+'.h5'
    yopopath = os.path.join(model_path,'deltamodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
    #gpath = os.path.join(model_path,'gmmumodel_'+'_'.join(model_name.split('_')[1:]))+'.h5'
    figure_path = os.path.join(figures_path,'_'.join(model_name.split('_')[1:]))
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    
    yopo = YOLO_Model()
    classmodel = YOPO_classification(num_labels=19, vector_size = 32)
    parallel_model = CombinedModel(yopo,classmodel)
    yopo.to(device)
    classmodel.to(device)

    yopo_stat = torch.load(yopopath)
    class_stat = torch.load(classmodelpath)
    yopo.load_state_dict(yopo_stat)
    classmodel.load_state_dict(class_stat)
    scanning_model = CombinedModel(yopo,classmodel)


    x_train = torch.tensor(x_train,dtype=torch.float32).to(device)    # [197,24,24,24,1]
    x_train = x_train.permute(0,4,1,2,3)  
    features = yopo(x_train)

    # you can set replacce args.candidateKs with some K
    labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = \
            statistical_fitting_split_merge(features = np.squeeze(features), \
                                               labels = None, candidateKs = args.candidateKs,\
                                                    K = None, reg_covar = args.reg_covar, it = 0,\
                                                    u_filter_rate=args.u_filter_rate, alpha = args.alpha)
    labels_soft = labels_temp_proba
    labels = labels_temp





    # labels = array([0,1,2,3,4,....])
    ############################################################
    ############################################################
    ############################################################
    for i in tqdm(range(np.max(labels) + 1)):
        locs = np.array(infos)[labels == i]
        v_i = np.zeros_like(v)
        for j in locs:
            # print("j=",j)
            if j[0] == 'emd_4603.map': #emd_4603_deconv_corrected.mrc / emd_4603.map
                v_i[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s] = \
                v[j[2][0] - s: j[2][0] + s, j[2][1] - s: j[2][1] + s, j[2][2] - s: j[2][2] + s]
        save_png(cub_img(v_i[:,:,::15])['im'], os.path.join(figure_path, 'GMMU'+str(i) + model_name + '.png'))
        # save_png(cub_img(v_i[:,:,:])['im'], os.path.join(figure_path, 'GMMU'+str(i) + model_name + '.png'))


'''