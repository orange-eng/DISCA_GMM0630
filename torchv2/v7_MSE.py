# 2024/7/12
# without scanning
# Chengzhi Cao

# v4将yaml文件里面的超参数都分开写进代码了，方便调试.
# v4使用了crossEntropyLoss,删除了MSELoss
# v4接下来需要加上positive和negative的数据进行对比训练,使用nn.CosineSimilarity
# v5加入了scanning部分，对每次iteration的数据进行修改
# v6使用softmax之后，再看看效果.并且及时保留了模型。之前训练结束之后都没有及时保留模型，是个失误
# v6并且对数据进行了归一化，方便训练一些，看看结果

import os, h5py, math
from tqdm import *
import sys
sys.path.append('/code/DISCA_GMM')
from disca_dataset.DISCA_visualization import *
# from hist_filtering.filtering import *
from config import *
from torch_DISCA_gmmu2 import *

from GMMU.torch_gmmu_cavi_stable_new import TORCH_CAVI_GMMU as GMM

import numpy as np
import sys, multiprocessing, importlib
from multiprocessing.pool import Pool

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# from GMMU.gmmu_cavi_stable_new import CAVI_GMMU as GMM
# from deepdpm_gmmu.split_merge_function_new import *                               

import gc

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch.optim as optim
import skimage,os

import faulthandler
faulthandler.enable()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
#dataset options
parser.add_argument("--filtered_data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/data.h5')
parser.add_argument("--data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/original')

#stored path
parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results7')
parser.add_argument("--algorithm_name", type=str, default='gmmu_cavi_llh_hist')
parser.add_argument("--filtered_particle_saving_path", type=str, default='/data/zfr888/EMD_4603/Results7/filtered_particle')

parser.add_argument("--image_size", type=int, default=24)
parser.add_argument("--input_size", type=int, default=24)
parser.add_argument("--candidateKs", default=[7,8,9,10,11])

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--factor", default=2)
parser.add_argument("--lr", default=0.01)
parser.add_argument("--momentum", default=0.9)
parser.add_argument("--loss_function",type=int, default=2)
parser.add_argument("--optimizer", default='adam')
parser.add_argument("--hidden_num",type=int, default=32)

parser.add_argument("--reg_covar", type=int, default=0.000001)
parser.add_argument("--u_filter_rate", type=str, default=0.025)
parser.add_argument("--alpha",type=float, default=1.0)

parser.add_argument("--scanning_bottom",type=int, default=100)
parser.add_argument("--scanning_upper",type=int, default=20000)
parser.add_argument("--num_labels",type=int, default=10)

parser.add_argument("--scanning_num", type=int, default=1)
parser.add_argument("--DIVIDE", type=int,default=10)
parser.add_argument("--M", type=int,default=1)
parser.add_argument("--sub_epoch", type=int,default=2)
parser.add_argument("--subtomo_num",type=int, default=1000)
parser.add_argument("--subtomo_num_test",type=int, default=100)
args = parser.parse_args()


def update_beta(beta_0, Nk):
    lambda_beta = beta_0 + Nk
    return lambda_beta

def update_nu(nu_0, Nk):
    lambda_nu = nu_0 + Nk
    return lambda_nu


def update_w_inv(responsibility, w_o, beta_o, m_o, x):

    d = x.shape[-1]
    Nk = torch.sum(responsibility, axis=0) #shape:([])
    xbar = torch.matmul(torch.unsqueeze(responsibility, axis=0), x.type(torch.float64))/ Nk #(d,)
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
    C = np.log(np.pi) * p * (p-1) /4
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
    lml = - N*D/2*np.log(np.pi)
    lml = torch.tensor(lml, dtype=x.dtype)
    beta_0 = torch.tensor(beta_0, dtype=x.dtype)
    w_0 = torch.tensor(w_0, dtype=x.dtype)
    nu_0 = torch.tensor(nu_0, dtype=x.dtype)
    lambda_beta = torch.tensor(lambda_beta, dtype=x.dtype)
    lambda_w_inv = torch.tensor(lambda_w_inv, dtype=x.dtype)
    lambda_nu = torch.tensor(lambda_nu, dtype=x.dtype)

    lml = lml + multi_log_gamma_function(lambda_nu/2, D) - multi_log_gamma_function(nu_0/2, D)
    lml = lml + nu_0/2*torch.log(torch.linalg.det(torch.linalg.inv(w_0))) - \
        lambda_nu/2*torch.log(torch.det(lambda_w_inv))
    lml = lml + (D/2)*(torch.log(beta_0)-torch.log(lambda_beta))
    return lml



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

    # alpha = torch.tensor(alpha, x.dtype)
    alpha = torch.tensor(alpha).type(x.dtype).to(device)

    prob_c_original = torch.sum(torch.log(torch.range(1, N, dtype=x.dtype))).to(device)
    prob_c_split1 = torch.sum(torch.log(torch.range(1, n_sub1, dtype=x.dtype))).to(device)
    prob_c_split2 = torch.sum(torch.log(torch.range(1, n_sub2, dtype=x.dtype))).to(device)

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
    
    # _alpha = torch.log(alpha)
    log_acceptpro = torch.log(alpha) + prob_c_split1 + prob_c_split2 - prob_c_original \
                    + sub1_log_prob+sub2_log_prob-cluster_split_log_prob \
                    + torch.tensor((n_sub1+n_sub2-2)*torch.log(torch.tensor([2.])), dtype=x.dtype).to(device)
    if np.isnan(log_acceptpro.cpu().detach().numpy()):
        if np.isnan(sub1_log_prob.cpu().detach().numpy()):
            print('sub1_log_prob',sub1_log_prob)
        if np.isnan(sub2_log_prob.cpu().detach().numpy()):
            print('sub2_log_prob',sub2_log_prob)
        if np.isnan(cluster_split_log_prob.cpu().detach().numpy()):
            print('cluster_split_log_prob',cluster_split_log_prob)


    return torch.exp(log_acceptpro)



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



def statistical_fitting_split_merge(features, labels, candidateKs, K, reg_covar, it, u_filter = True,u_filter_rate=0.0025, alpha = 1.0):

    features_pca = features
    n_features = features_pca.shape[1]

    reg_covar = np.max([reg_covar * (0.5**it), 1e-6])    
    labels_K = [] 
    models = []
    BICs = [] 
    
    # K = 2
    k_new = K
    k_original = K

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    if K in candidateKs:
        gmm_0 = GMM(n_cluster = K, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate, \
                    reg_covar=reg_covar)
        gmm_0 = gmm_0.to(device)
        gmm_0.fit(features_pca) 
        labels_k_0 = gmm_0.hard_assignment()
        alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = gmm_0.parameters_o()
        r_cluster = gmm_0.soft_assignment()
        r_cluster[0] = r_cluster[1]
        hardcluster_label = gmm_0.hard_assignment()


        for i in range(1, K+1): #ignore uniform
            x_i = features_pca[hardcluster_label == i,:]
            if x_i.shape[0] <= 1 :
                    continue
            gmm_i = GaussianMixture(n_components=2, covariance_type='full', \
                                    tol=0.001, random_state=i, init_params = 'kmeans', reg_covar=reg_covar)
            # gmm_i = gmm_i.to(device)
            gmm_i.fit(x_i.cpu().detach().numpy()) 
            subcluster_i_hard_label = gmm_i.predict(x_i.cpu().detach().numpy())
            x_i1 = x_i[subcluster_i_hard_label==0, :]
            x_i2 = x_i[subcluster_i_hard_label==1, :]
            if x_i1.shape[0] == 0 or x_i2.shape[0] == 0:
                    continue
            r_all = r_cluster[:, i]  # (n, )
            r_sub = torch.unsqueeze(r_all, axis=-1) * torch.tensor(gmm_i.predict_proba(features_pca.cpu().detach().numpy())).to(device)  # (n, 2)
            r_c1 = r_sub[:, 0]  # (n, )
            r_c2 = r_sub[:, 1]  # (n, )

            if k_new >= np.max(candidateKs):
                break
            else:
                k_new = k_new + 1

        if k_new != K:
            gmm_1 = GMM(n_cluster = k_new, a_o = None, b_o = None, u_filter = u_filter, \
                        u_filter_rate = u_filter_rate, reg_covar=reg_covar)
            gmm_1 = gmm_1.to(device)
            gmm_1.fit(features_pca)
        else:
            gmm_1 = gmm_0

        alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = gmm_1.parameters_o()
        lambda_alpha, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o = gmm_1.parameters()
        softcluster_label = gmm_1.soft_assignment() #tensor
        hardcluster_label = gmm_1.hard_assignment() #numpy
        k_new2 = k_new

        if k_new2 != k_new:
            gmm = GMM(n_cluster = k_new2, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate, reg_covar=reg_covar)
            gmm = gmm.to(device)
            gmm.fit(features_pca)
        else:
            gmm = gmm_1
        
        labels_k = gmm.hard_assignment()
        K_temp = k_new2 #ignore uniform 


    else: 
        K_temp = int(0.5*(np.min(candidateKs) + np.max(candidateKs)))
        gmm = GMM(n_cluster = K_temp, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate,reg_covar=reg_covar) 
        gmm = gmm.to(device)
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
            
            # print('label=',labels)
            # print("K+1=",K+1)
            print("features_pca_433=",features_pca)
            precisions_init = np.array(
                [np.linalg.pinv(np.cov(features_pca[labels == j].T.cpu().detach().numpy())) for j in range(1,K+1)])
            
            gmm = GMM(n_cluster=K, a_o=None, b_o=None, u_filter=u_filter, u_filter_rate=u_filter_rate, \
                    reg_covar=reg_covar, init_param="self_setting",\
                    weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)
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


    print('Estimated K:', K)
    return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm   






def con_hist_filtering(X, scanning_bottom=100, scanning_upper=20000,saving_path = None):
    """
    X: the input data (w,h,l)
    saving_path: saving the path of each labels components
    neighbor_size: the neighbor arrays that are counted
    """
    dic_components_nums = {}
    output_index = []
    for label_i in np.unique(X):
        if label_i == 0:
            #print('remember to delete, label 0!')
            background_index = np.array(np.where(X == label_i)).T
            output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], int(0.1*np.prod(X.shape)))),])
            #dic_components_nums[label_i] = components_nums
            continue

        # con_nums doesn't include the background 0
        con_labels, con_nums = skimage.measure.label(np.where(X == label_i, X, 0), return_num=True)
        #print('remember to delete',np.unique(con_labels), con_nums)
        components_nums = []
        for i in range(con_nums):
            components_nums.append(np.sum(con_labels == i+1))
        #
        dic_components_nums[label_i] = components_nums
        #

        qts_bottom = np.min([np.quantile(components_nums, 0.05),scanning_bottom]) # np.quantile(components_nums, 0.25)
        qts_up = np.min([np.quantile(components_nums, 0.85),scanning_upper]) #is the maximum 20000?
        for i in range(con_nums):
            if np.sum(con_labels == i+1) >= qts_bottom and np.sum(con_labels == i+1) <= qts_up:
                output_index.append(np.floor(np.mean(np.array(np.where(con_labels == i+1)).T, axis=0).reshape([1, -1])))
    
    if len(output_index)==0:
        background_index = np.array(np.where(X != 0)).T
        output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], 10)),])
    output_index = np.concatenate(output_index)

    if saving_path is not None:
        np.save(saving_path, dic_components_nums)

    return output_index



def data_augmentation_simple(x_train, factor = 2):
    '''
    rotation, smooth, 挖空
    '''
    if factor > 1:

        x_train_augmented = []
        
        image_size = x_train.shape[1]
        
        x_train_augmented.append(x_train)

        for f in range(1, factor):
            ts = {}        
            for i in tqdm(range(len(x_train))): #batch size                      
                v = x_train[i,:,:,:,0]
                Inv_R = random_rotation_matrix()
                sigma = np.random.uniform(0, 2.0) 
                alpha = np.random.uniform(0.8, 1.2)  
                beta = np.random.uniform(-0.2, 0.2) 
                start = np.random.randint(0, image_size, 3)
                end = start + np.random.randint(0, image_size/4, 3)
                x_train_f = augment(v, Inv_R, sigma, alpha, beta, start, end)
                x_train_f = np.expand_dims(np.expand_dims(x_train_f, -1), 0)
                x_train_augmented.append(x_train_f)
            
        x_train_augmented = np.concatenate(x_train_augmented)
    
    else:
        x_train_augmented = x_train                        

    return x_train_augmented

def prepare_training_data_simple(x_train, labels_temp_proba, labels, n):
    # n代表着重复生成data的次数
    # prevent changes on the x_train and labels
    x_train=x_train.copy()
    labels_temp_proba=labels_temp_proba.copy()
    labels=labels.copy()

    label_one_hot = one_hot(labels, len(np.unique(labels))) 
     
    index = np.array(range(x_train.shape[0] * n))
    
    labels_tile = np.tile(labels, n) # m*n
    
    labels_proba_tile = np.tile(labels_temp_proba, (n, 1)) # (m*n)*n m*n的组合在列重复n次
    
    labels_np = []
    
    for i in range(len(np.unique(labels))):
        npi = np.maximum(0, 0.5 - labels_proba_tile[:, i][labels_tile != i]) # 逐元素比较， 非 i th cluster的样本第i个cluster的概率
        
        labels_np.append(npi/np.sum(npi))
    
    #because n=1 here, it will not run like pos or negative
    x_train_augmented = data_augmentation_simple(x_train, n)
    x_train_augmented_pos = data_augmentation_simple(x_train, n + 1)[x_train.shape[0]:]
    
    #index 抽样
    index_negative = np.array([np.random.choice(a = index[labels_tile != labels_tile[i]], p = labels_np[labels_tile[i]]) for i in range(len(index))])
    x_train_augmented_neg = data_augmentation_simple(x_train, n + 1)[x_train.shape[0]:][index_negative]

    np.random.shuffle(index)              
    #x_train_permute = [x_train_augmented[index].copy(), x_train_augmented_pos[index].copy(), x_train_augmented_neg[index].copy()]
    x_train_permute = [x_train_augmented[index], x_train_augmented_pos[index], x_train_augmented_neg[index]]  
    #labels_permute = [np.tile(label_one_hot, (n, 1))[index].copy(), np.tile(label_one_hot, (n, 1))[index].copy(), np.tile(label_one_hot, (n, 1))[index_negative][index].copy()]
    labels_permute = [np.tile(label_one_hot, (n, 1))[index], np.tile(label_one_hot, (n, 1))[index], np.tile(label_one_hot, (n, 1))[index_negative][index]] 

    return label_one_hot, x_train_permute, labels_permute
    

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################




# data set
filtered_data_path = args.filtered_data_path
h5f = h5py.File(filtered_data_path,'r')                                                        
total_subtomo = len(h5f['dataset_1'][:]) # only 'dataset_1'  [16265,24,24,24,24] 

subtomo_num = args.subtomo_num
filtered_data = h5f['dataset_1'][total_subtomo- subtomo_num:] # only 'dataset_1'  [16265,24,24,24,24]                            

h5f.close()
data_path = args.data_path

# setting of YOPO and GMMU
image_size = args.image_size #None   ### subtomogram size ###
input_size = args.input_size
candidateKs = args.candidateKs   ### candidate number of clusters to test
        
BATCH_SIZE = args.batch_size
scanning_num = args.scanning_num ### the number of scanning ###
factor = args.factor ### the num of scanning division###
M = args.M   ### number of iterations ###
LR = args.lr   ### CNN learning rate ###
lr = args.lr   ### CNN learning rate ###
DIVIDE = args.DIVIDE
sub_epochs = args.sub_epoch
reg_covar = args.reg_covar

hidden_num = args.hidden_num

# paths used to stored
saving_path = args.saving_path
algorithm_name = args.algorithm_name
model_path = saving_path+'/models/deltamodel_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))
classification_model_path = saving_path+'/models/classificationmodel_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))

combined_model_path = saving_path+'/models/combinedmodel_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))

gmmu_model_path = saving_path+'/models/gmmumodel_%s_M_%s_epoch_%s_lr_%s_reg_%s.h5' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))
model_path_last = saving_path+'/models/deltamodel_%s_M_%s_epoch_%s_lr_%s_reg_%s_last.h5' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))

label_path = saving_path+'/results/labels_%s_M_%s_epoch_%s_lr_%s_reg_%s.pickle' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))
label_path_last = saving_path+'/results/labels_%s_M_%s_epoch_%s_lr_%s_reg_%s_last.pickle' \
    %(algorithm_name,str(M),str(sub_epochs),str(lr),str(reg_covar))

for creat_path in ['/models','/figures','/results']:
    creat_folder_path = saving_path+creat_path
    if not os.path.exists(creat_folder_path):
        os.makedirs(creat_folder_path)
 

### Generalized EM Process ### 

pp_indexs = []

for scanning_it in range(scanning_num):
    print('# Scanning:', scanning_it)

    ### Load data ###   
    DBI_best = np.inf
    K = None
    LR = args.lr
    labels = None 
    it = 0
    best_i = it
    done = False
    x_train = []    
    fi = 0    
    for f in sorted(os.listdir(data_path)):   
        if f.split("_")[0] != 'emd':
            continue 
        tom = read_mrc_numpy_vol(os.path.join(data_path,f))  # tom.shape=[928,928,464]
        tom = (tom - np.mean(tom))/np.std(tom)      
        tom[tom > 4.] = 4.    
        tom[tom < -4.] = -4.
        # the first training, you have to randomly cut subtomo from tom as x_train data
        if scanning_it < 10:                                    
            n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2),\
                            np.random.randint(input_size/2,tom.shape[1]-input_size/2),\
                            np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(subtomo_num)])
        else:        
        # if not first training, where are some randomly init, and some generated data
            n = np.array([[np.random.randint(input_size/2,tom.shape[0]-input_size/2), \
                            np.random.randint(input_size/2,tom.shape[1]-input_size/2), \
                            np.random.randint(input_size/2,tom.shape[2]-input_size/2)] for pi in range(subtomo_num//2)]) 

            n = np.concatenate([n, pp_indexs[fi][np.random.randint(pp_indexs[fi].shape[0], size=subtomo_num//2), :] + input_size/2])

        for j in range(len(n)): #random cutting from tomo, time: 0.43s
            v = cut_from_whole_map(tom, n[j], input_size)
            if v is not None:
                x_train.append(v)
            
        fi += 1
                    
    x_train = np.expand_dims(np.array(x_train), -1)  
    x_train_min = np.min(x_train, axis = (1,2,3,4))           
    # x_train = x_train[x_train_min < np.median(x_train_min)] # [95,24,24,24,1]
    x_train = np.concatenate([x_train, filtered_data])  # [153,24,24,24]
    # x_train = filtered_data  # [153,24,24,24]
    # x_train=[50,24,24,24,1]



    total_num = len(x_train)//DIVIDE


    _iter = 0
    while not done:     # done=False, it will continue iteration 
        yolo_model = YOLO_Model(hidden_channel=args.hidden_num)
        print('## Iteration:', it) 
        ### Feature Extraction ### 
        if it != 0: 
            yolo_state_dict = torch.load(model_path)
            yolo_model.load_state_dict(yolo_state_dict)


        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yolo_model.to(device)
        # 优化器和损失函数
        criterion = nn.CrossEntropyLoss(reduction='mean')
        criterion = criterion.cuda()
        optimizer = optim.SGD(yolo_model.parameters(),lr = LR,momentum=args.momentum)

        total_feat = torch.tensor([])

        # yolo_model.eval()
        with torch.no_grad():

            for a in range(total_num):
                # print("a=",a)
                x_train2 = x_train[a*DIVIDE : a*DIVIDE+DIVIDE] # only 'dataset_1'                              
                x_train2 = torch.tensor(x_train2,dtype=torch.float32)
                x_train2 = x_train2.to(device)    # [197,24,24,24,1]
                x_train2 = x_train2.permute(0,4,1,2,3)  
                features = yolo_model(x_train2)

                if a == 0:
                    total_feat = features
                else:
                    total_feat = torch.cat((total_feat,features))


        ## Feature Clustering ###                              
        # labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = statistical_fitting_split_merge(features = np.squeeze(features), \
        #         labels = labels, candidateKs = candidateKs,\
        #         K = K, reg_covar = reg_covar, it = it,\
        #         u_filter_rate=args.u_filter_rate, alpha = args.alpha)

        features = total_feat
        features = np.squeeze(features)
        u_filter_rate=args.u_filter_rate
        alpha = args.alpha
        u_filter = True
        # u_filter_rate=0.0025, alpha = 1.0
        ##############################################################################################
        ##############################################################################################
        ##############################################################################################

        features_pca = features
        n_features = features_pca.shape[1]

        reg_covar = np.max([reg_covar * (0.5**it), 1e-6])    
        labels_K = [] 
        models = []
        BICs = [] 
        
        # K = 2
        k_new = K
        k_original = K
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        K_temp = int(0.5*(np.min(candidateKs) + np.max(candidateKs)))
        gmm = GMM(n_cluster = K_temp, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate,reg_covar=reg_covar) 
        gmm = gmm.to(device)
        gmm.fit(features_pca)       # features_pca=[20,32]
        labels_k = gmm.hard_assignment()    # give the label of each features, labels_k=

        same_K = False 
        K = K_temp

        labels_temp = remove_empty_cluster(labels_k)
        labels_temp_proba = gmm.soft_assignment().cpu().numpy()

        print('Estimated K:', K)
        ###########################################################
        ###########################################################
        ###########################################################


        ### Matching Clusters by Hungarian Algorithm ### 
        # if same_K: 
        #     labels_temp, col_ind = align_cluster_index(labels, labels_temp)
        #     labels_temp_proba = labels_temp_proba[:,col_ind]

        # index is changed, but they represent the same label

        labels_temp[0] = labels_temp[1] # 第一维度出现很多nan，没法继续往下走
        labels_temp_proba[0] = labels_temp_proba[1] # 第一维度出现很多nan，没法继续往下走
        # i, the numb of iteration, is added 1 here
        it, labels, done = convergence_check(i = it, M = M, labels_temp = labels_temp, labels = labels, done = done) 

        if it == M: 
            done = True
       
        print('set(labels)=',set(labels))# set() 将集合中重复的元素去除
        print('## Cluster sizes:', [np.sum(labels == k) for  k in set(labels)])  # 返回每个标签的数量


        ### Validate Clustering by distortion-based DBI ### 
        # depending the intialization, DBI could be Nan            
        DBI = DDBI_uniform(features_pca, labels) 
        # DBI = nan 无限小，DBI_best = inf无限大
        if np.isnan(DBI):
            DBI = 1e-6
        

        _unique = np.unique(labels)
        
        # model_classification = YOPO_classification(num_labels=len(set(labels)), vector_size = 32) 
        model_classification = YOPO_classification(num_labels=len(np.unique(labels)), vector_size = 32)  
        model_classification = model_classification.to(device)
        # 模型存储
        if DBI < DBI_best: 
            if it > 1:             
                # 保存整个模型
                torch.save(yolo_model.state_dict(), model_path)
                torch.save(model_classification.state_dict(), classification_model_path)

                labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
                pickle_dump(labels_best, label_path)
                print('## new model_{} is saved'.format(it)) 
                best_i = it
            # else: 
            #     torch.save(yolo_model.state_dict(), model_path)          
            #     torch.save(model_classification.state_dict(), classification_model_path)
            #     labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
            #     pickle_dump(labels_best, label_path)
            #     print('## new modele is saved') 
            #     best_i = it
            DBI_best = DBI                                             
        print('## DDBI:', DBI)
        if np.isinf(DBI) and it>1:
            continue 

        ### Permute Samples ###             
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'Preparing data sets')
        # prepare_training_data_simple doesn't use the parallel acceleration like prepare_training_data
        if np.any(np.isnan(labels_temp_proba)) or np.any(np.isnan(labels)):
            if np.any(np.isnan(labels_temp_proba)) and np.any(np.isnan(labels)):
                print('there is nan in labels_temp_proba and labels')
                continue
            if np.any(np.isnan(labels)):
                print('there is nan in labels')
                continue
            if np.any(np.isnan(labels_temp_proba)):
                print('there is nan in labels_temp_proba')
                continue
        # label_one_hot, x_train_permute, labels_permute = \
        #     prepare_training_data_simple(x_train = x_train, labels_temp_proba = labels_temp_proba, labels = labels, n = 1) 

        label_one_hot, x_train_permute, labels_permute = prepare_training_data(x_train = x_train, labels_temp_proba = labels_temp_proba, labels = labels, n = 1)


        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Preparing data sets ends')

        features = features.cpu().detach().numpy()
        features = np.squeeze(features)

        _features = [features, features, features] # _features 有三个向量，每个都是(15,32)
        _label = [label_one_hot, label_one_hot, label_one_hot]


        _features = torch.tensor(features)
        _label = torch.tensor(labels)
        class_dataset = Data.TensorDataset(_features, _label)  

        classification_dataloader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=class_dataset,       
            batch_size=BATCH_SIZE,       
            shuffle=True,                
            num_workers=0,               
        )

        ### Finetune new model with current estimated K ### 
        if not same_K:                                      
                
            optimizer = optim.SGD(model_classification.parameters(),lr = LR,momentum=0.5)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            criterion = criterion.cuda()
            for epoch in range(args.sub_epoch):
                for i, data in enumerate(classification_dataloader, 0):
                    input, target = data    # input=[2,4,32], target=[2,4]
                    input, target = input.to(device),target.to(device)
                    y_pred = model_classification(input)    # y_pred=[2,4]
                    loss = criterion(y_pred, target)
                    # print(i+1,epoch+1,loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        ### New YOPO ###
        # same reason as squeeze before to use flatten
        submodel1 = YOLO_Model()
        submodel2 = YOPO_classification(num_labels=len(np.unique(labels)), vector_size = 32)
        parallel_model = CombinedModel(submodel1,submodel2)
        parallel_model = parallel_model.to(device)


        ### CNN Training ###   
        # add scope?        
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Train new model')
        lr *= 0.95 
                
        # optimizer = optim.SGD(model_classification.parameters(),lr = LR,momentum=0.5)
        optimizer = optim.SGD(parallel_model.parameters(),lr = LR,momentum=0.5)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        MSE_criterion = nn.MSELoss()
        criterion = criterion.cuda()
        
        _x_train_permute0 = torch.unsqueeze(torch.tensor(x_train_permute[0]),1)
        _x_train_permute1 = torch.unsqueeze(torch.tensor(x_train_permute[1]),1)
        _x_train_permute2 = torch.unsqueeze(torch.tensor(x_train_permute[2]),1)
        _x_train_permute_total = torch.cat((_x_train_permute0,_x_train_permute1,_x_train_permute2),dim=1)
        _x_train2 = torch.tensor(_x_train_permute_total,dtype=torch.float32)

        x_max = torch.max(_x_train2)
        x_min = torch.min(_x_train2)
        print('x_max=',x_max)
        print('x_min=',x_min)

        _x_train2 = (_x_train2 - x_min)/(x_max - x_min)

        x_max = torch.max(_x_train2)
        x_min = torch.min(_x_train2)
        print('x2_max=',x_max)
        print('x2_min=',x_min)
        ###################################################################################
        ###################################################################################
        ###################################################################################

        _labels_permute0 = torch.unsqueeze(torch.tensor(np.argmax(labels_permute[0],axis=1)),1)
        _labels_permute1 = torch.unsqueeze(torch.tensor(np.argmax(labels_permute[1],axis=1)),1)
        _labels_permute2 = torch.unsqueeze(torch.tensor(np.argmax(labels_permute[2],axis=1)),1)

        # _labels_permute_total = torch.cat((_labels_permute0,_labels_permute1,_labels_permute2),dim=1)
        # x_label = torch.tensor(_labels_permute_total,dtype=torch.long)
        label_one_hot = torch.unsqueeze(torch.tensor(label_one_hot,dtype=torch.float),2)
        _labels_permute_total = torch.cat((label_one_hot,label_one_hot,label_one_hot),dim=2)
        x_label = torch.tensor(_labels_permute_total)

        torch_dataset2 = Data.TensorDataset(_x_train2, x_label)  # 对给定的 tensor 数据，将他们包装成 dataset
        train_dataloader2 = Data.DataLoader(
            dataset=torch_dataset2,       
            batch_size=BATCH_SIZE,       
            shuffle=True,                
            num_workers=0,               
        )

        for epoch in range(args.sub_epoch):
            for i, data in enumerate(train_dataloader2, 0):

                _input, target = data    # input=[2,4,24,24,24,1],target=[2,4]
                _input, target = _input.to(device),target.to(device)

                input_normal = _input[:,0]
                input_pos = _input[:,1]
                input_neg = _input[:,2]
                _input_normal = input_normal.permute(0,4,1,2,3)
                _input_pos = input_pos.permute(0,4,1,2,3)
                _input_neg = input_neg.permute(0,4,1,2,3)

                target_normal = target[:,:,0]
                target_pos = target[:,:,1]
                target_neg = target[:,:,2]
                
                y_pred_normal = parallel_model(_input_normal)
                y_pred_pos = parallel_model(_input_pos)
                y_pred_neg = parallel_model(_input_neg)
                # print('_input_normal=',_input_normal[0][0][0][0])
                
                # print('target_normal=',target_normal.shape)
                # print('target_normal=',target_normal)
                pred_labels = np.array([np.argmax(y_pred_normal[q, :].cpu().detach().numpy()) for q in range(len(y_pred_normal))])
                print('pred_labels=',pred_labels)


                target_labels = np.array([np.argmax(target_normal[q, :].cpu().detach().numpy()) for q in range(len(target_normal))])
                # print('target_labels=',target_labels)
                
                # cos = nn.CosineSimilarity()
                # cos1 = cos(_input_normal, _input_pos)
                # cos2 = cos(_input_normal, _input_neg)
                # zeros = torch.zeros(cos1.shape,device=device)

                # mse_loss1 = MSE_criterion(cos1,zeros)
                # mse_loss2 = MSE_criterion(cos2,zeros)
                loss1 = MSE_criterion(y_pred_normal, target_normal)
                # loss2 = criterion(y_pred_pos, target_pos)
                # loss3 = criterion(y_pred_neg, target_neg)
                # loss = loss1 + loss2 + loss3 + mse_loss1 + mse_loss2
                loss = loss1
                
                if i % 1 == 0:
                    print("i={},iter={},epoch={},loss={}".format(i+1,_iter+1,epoch+1,loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        _iter = _iter + 1

        if _iter == 1:             
            torch.save(submodel1.state_dict(), model_path)          
            torch.save(submodel2.state_dict(), classification_model_path)
            torch.save(parallel_model.state_dict(), combined_model_path)
            # labels_best = labels.copy()   ### save current labels if DDBI improves ###                 
            # pickle_dump(labels_best, label_path)
            print('## new model_{} is saved'.format(_iter)) 
    #print('## DDBI:', DBI)
    print('## the best iteration is %s' % str(best_i))



    ############################################################################################
    ############################################################################################
    ############################################################################################
    ############################################################################################
    ### scanning ###  
    # building the subtomo #   

    '''
    parallel_model_feature = torch.load(model_path)
    model_classification = torch.load(classification_model_path)

    submodel1.load_state_dict(parallel_model_feature)
    submodel2.load_state_dict(model_classification)

    scanning_model = CombinedModel(submodel1,submodel2)




    fi = 0    
    pp_indexs = []
    for f in sorted(os.listdir(data_path)):   
        if f.split("_")[0] != 'emd':
            continue 
        tom = read_mrc_numpy_vol(os.path.join(data_path,f))            
        tom = (tom - np.mean(tom))/np.std(tom)        
        tom[tom > 4.] = 4.    
        tom[tom < -4.] = -4.   
        adding_pre = math.floor(input_size/2)       # adding_pre=12
        adding_post = math.ceil(input_size/2)-1     # adding_post=11
        # tom.shape=[928,928,464], factor=10
        # x_interval_start, y_interval_start, z_interval_start = \
        #     [np.array(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor))) for i in range(3)]                        

        # x_interval_end, y_interval_end, z_interval_end = \
        #     [np.array(list(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor)))[1:] + [tom.shape[i]]) for i in range(3)]   
            
        # x_interval_start -= adding_pre
        # y_interval_start -= adding_pre
        # z_interval_start -= adding_pre
        # x_interval_end[:-1] += adding_post
        # y_interval_end[:-1] += adding_post
        # z_interval_end[:-1] += adding_post   

        # subvolumes = []        
        # #print('interval num: ', len(x_interval_start)) 

        # for i in range(factor): 
        #     for j in range(factor):           
        #         for k in range(factor):       
        #             subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], \
        #                             z_interval_start[k]: z_interval_end[k]]    
        #             subvolumes.append(np.expand_dims(np.array(subvolume), [0,-1]))


        # x_interval_start = np.linspace(0, 888, num=38).astype(int)
        # x_interval_end = np.linspace(24, 912, num=38).astype(int)

        # y_interval_start = np.linspace(0, 888, num=38).astype(int)
        # y_interval_end = np.linspace(24, 912, num=38).astype(int)

        # z_interval_start = np.linspace(0, 432, num=19).astype(int)
        # z_interval_end = np.linspace(24, 456, num=19).astype(int)


        x_interval_start = np.linspace(0, 72, num=4).astype(int)
        x_interval_end = np.linspace(24, 96, num=4).astype(int)

        y_interval_start = np.linspace(0, 72, num=4).astype(int)
        y_interval_end = np.linspace(24, 96, num=4).astype(int)

        z_interval_start = np.linspace(0, 72, num=4).astype(int)
        z_interval_end = np.linspace(24, 96, num=4).astype(int)

        subvolumes = []
        for i in range(len(x_interval_start)): 
            for j in range(len(y_interval_start)):           
                for k in range(len(z_interval_start)):       
                    subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], \
                                    z_interval_start[k]: z_interval_end[k]]    
                    subvolumes.append(np.expand_dims(np.array(subvolume), [0,-1]))
        # subvolumes = subvolumes[0:20]
        # predict #
        subvolumes_label = []
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'constructing new data sets:')
        for subv_i in tqdm(range(len(subvolumes))):
            subt = subvolumes[subv_i]           # # subt=[1,115,115,69,1]
            subt = torch.tensor(subt)
            subt = subt.permute(0,4,1,2,3)
            subt_label = scanning_model(subt) # subt=[1,1,46,46,34],sub_label=[1,4]
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.squeeze(subt_label,dim=0)
            subt_label = subt_label.detach().numpy()
            subt_label = np.max(subt_label,axis=0)
            
            subvolumes_label.append(subt_label)

        pp_map = np.zeros([tom.shape[0] - (input_size - 1), \
                        tom.shape[1] - (input_size - 1), \
                            tom.shape[2] - (input_size - 1), \
                                subvolumes_label[0].shape[-1]]) # pp_map.shape=[905,905,441,4]
        m = 0                 
        for i in tqdm(range(len(x_interval_start))):             
            for j in range(len(y_interval_start)):             
                for k in range(len(z_interval_start)):
                    pp_map[x_interval_start[i]: x_interval_start[i] + subvolumes_label[m].shape[1], \
                        y_interval_start[j]: y_interval_start[j] + subvolumes_label[m].shape[2], \
                            z_interval_start[k]: z_interval_start[k] + subvolumes_label[m].shape[3]] = subvolumes_label[m]   
                    m += 1


        #pp_map_filtered_labels = np.where(pp_map[:, :, :, 0]<0.5,np.argmax(pp_map, -1),0) # (l,w,h)
        pp_map_filtered_labels = np.argmax(pp_map, -1)


        particle_filtered = []
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'scanning:')
        
        x_con_start, y_con_start, z_con_start = [np.array(range(0, pp_map_filtered_labels.shape[i], \
                    int(pp_map_filtered_labels.shape[i]/factor)))[:-1] for i in range(3)]
        x_con_end, y_con_end, z_con_end = [np.array(list(range(0, pp_map_filtered_labels.shape[i], \
                                        int(pp_map_filtered_labels.shape[i]/factor)))[1:-1] \
                            + [pp_map_filtered_labels.shape[i]]) for i in range(3)]



        for i in tqdm(range(len(x_interval_start))):
            for j in range(len(y_interval_start)):
                for k in range(len(z_interval_start)):
                    pp_subvolume = pp_map_filtered_labels[x_con_start[i]: x_con_end[i], \
                                y_con_start[j]: y_con_end[j], \
                                    z_con_start[k]: z_con_end[k]]
                    if args.filtered_particle_saving_path is None:
                        particle_filtered.append(con_hist_filtering(pp_subvolume,\
                        scanning_bottom=args.scanning_bottom, scanning_upper=args.scanning_upper))
                    else:
                        if not os.path.exists(args.filtered_particle_saving_path):
                            os.makedirs(args.filtered_particle_saving_path)
                        particle_filtered.append(con_hist_filtering(pp_subvolume,\
                        scanning_bottom=args.scanning_bottom, scanning_upper=args.scanning_upper,\
                        saving_path = '%s/hist_%s_%s_%s.npy' %(args.filtered_particle_saving_path,str(i),str(j),str(k))))

        pp_index = np.concatenate(particle_filtered)
        #save_png(cub_img(pp_map_non_noise[:, :, ::20])['im'], '/local/scratch/v_yijian_bai/disca/deepgmmu/disca/v.png')

        pp_indexs.append(pp_index)
        
        '''