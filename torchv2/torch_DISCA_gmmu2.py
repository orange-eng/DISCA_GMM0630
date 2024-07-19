import time, pickle
from tqdm import *
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import numpy as np
import scipy
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix 
from scipy.optimize import linear_sum_assignment

import sys, multiprocessing, importlib
from multiprocessing.pool import Pool
from skimage.transform import rescale 


from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA 
# from GMMU.gmmu_cavi_stable_new import CAVI_GMMU as GMM
# from deepdpm_gmmu.split_merge_function_new import *                               

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


Worker_Num = 4


def pickle_load(path): 
    with open(path, 'rb') as f:     o = pickle.load(f, encoding='latin1') 

    return o 



def pickle_dump(o, path, protocol=2):
    with open(path, 'wb') as f:    pickle.dump(o, f, protocol=protocol)


                                                                                    
def align_cluster_index(ref_cluster, map_cluster):                    
    """                                                        
    remap cluster index according the the ref_cluster.                
    both inputs must have same number of unique cluster index values.  
    """                                                               
                                                               
    ref_values = np.unique(ref_cluster)                               
    map_values = np.unique(map_cluster)                               
                                                                                    
    if ref_values.shape[0]!=map_values.shape[0]:               
        print('error: both inputs must have same number of unique cluster index values.')   
        print(ref_values,map_values)                                                                                                   
        return()                                           
    cont_mat = contingency_matrix(ref_cluster, map_cluster) #比较两个cluster的每个label的数量，制成n*n的表格                                                                                                                
    
    # linear_sum_assignment解决指派问题，返回最佳的行与列index
    # 用n减去每个entry就可以求解最佳的指派（两种cluster的最相近的label）                                                               
    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)                                                                                            
                                                                                    
    map_cluster_out = map_cluster.copy()                       
                                                                                    
    for i in ref_values:                                        
                                                                                    
        map_cluster_out[map_cluster == col_ind[i]] = i                 

    return map_cluster_out, col_ind


def align_cluster_index_uniform(ref_cluster, map_cluster):                    
    """                                                        
    remap cluster index according the the ref_cluster.                
    both inputs must have same number of unique cluster index values.  
    """                                                               
                                                               
    ref_values = np.unique(ref_cluster)                               
    map_values = np.unique(map_cluster)                               
                                                                                    
    if ref_values.shape[0]!=map_values.shape[0]:               
        print('error: both inputs must have same number of unique cluster index values.')                                                                                                      
        return()                                           
    cont_mat = contingency_matrix(ref_cluster, map_cluster) #比较两个cluster的每个label的数量，制成n*n的表格                                                                                                                
    
    # linear_sum_assignment解决指派问题，返回最佳的行与列index
    # 用n减去每个entry就可以求解最佳的指派（两种cluster的最相近的label）                                                               
    row_ind, col_ind = linear_sum_assignment(len(ref_cluster) - cont_mat)                                                                                            
                                                                                    
    map_cluster_out = map_cluster.copy()                       
                                                                                    
    for i in ref_values:                                        
                                                                                    
        map_cluster_out[map_cluster == col_ind[i]] = i                 

    return map_cluster_out, col_ind

    
    
def DDBI(features, labels):
    """
    Davies Bouldin index with different definition of the compact and distance between clusters
    """
    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = np.array([np.linalg.inv(np.cov(features[labels == i].T) + 1e-6 * np.eye(features.shape[1])) for i in np.unique(labels)])

    T = np.array([np.mean(np.diag((features[labels == i] - means_init[i]).dot(precisions_init[i]).dot((features[labels == i] - means_init[i]).T))) for i in np.unique(labels)])
    
    D = np.array([np.diag((means_init - means_init[i]).dot(precisions_init[i]).dot((means_init - means_init[i]).T)) for i in np.unique(labels)])
    
    DBI_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
    
    for i in range(len(np.unique(labels))):
        for j in range(len(np.unique(labels))):
            if i != j:
                DBI_matrix[i, j] = (T[i] + T[j])/(D[i, j] + D[j, i])
            
    DBI = np.mean(np.max(DBI_matrix, 0))
    
    return DBI 


def DDBI_uniform(features_o, labels_o):
    """
    Davies Bouldin index with different definition of the compact and distance between clusters
    In GMMU, the uniform cluster 0 is ignored
    """
    features = features_o[labels_o != 0]
    labels = labels_o[labels_o != 0]
    if labels.size == 0:
        return np.inf
    
    features = features.cpu().detach().numpy()
    # features2 = features.copy()
    means_init = np.array([np.mean(features[labels == i], 0) for i in np.unique(labels)])
    precisions_init = np.array([np.linalg.inv(np.cov(features[labels == i].T) + 1e-6 * np.eye(features.shape[1])) for i in np.unique(labels)])

    T = np.array([np.mean(np.diag((features[labels == i] - means_init[i-1]).dot(precisions_init[i-1]).dot((features[labels == i] - means_init[i-1]).T))) for i in np.unique(labels)])
    
    D = np.array([np.diag((means_init - means_init[i-1]).dot(precisions_init[i-1]).dot((means_init - means_init[i-1]).T)) for i in np.unique(labels)])
    
    DBI_matrix = np.zeros((len(np.unique(labels)), len(np.unique(labels))))

    for i in range(len(np.unique(labels))):
        for j in range(len(np.unique(labels))):
            if i != j:
                DBI_matrix[i, j] = (T[i] + T[j])/(D[i, j] + D[j, i])
            
    DBI = np.mean(np.max(DBI_matrix, 0))
    # if np.isnan(DBI):
    #     # print(np.unique(labels)) 
    #     # print(means_init)
    #     # print(precisions_init)
    #     print('DBI_matrix=',DBI_matrix)
    return DBI 




def run_iterator(tasks, worker_num=multiprocessing.cpu_count(), verbose=True):
    if verbose:		print('tomominer.parallel.multiprocessing.util.run_iterator()', 'start', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    worker_num = min(worker_num, multiprocessing.cpu_count())
    print('worker_num:', worker_num)

    for i,t in tasks.items():
        if 'args' not in t:     t['args'] = ()
        if 'kwargs' not in t:     t['kwargs'] = {}
        if 'id' not in t:   t['id'] = i
        assert t['id'] == i

    completed_count = 0 
    if worker_num > 1:

        pool = Pool(processes = worker_num)
        #pool = Pool()
        pool_apply = []
        for i,t in tasks.items():
            aa = pool.apply_async(func=call_func, kwds={'t':t})

            pool_apply.append(aa)

        if verbose:
            print('start getting results')
        for pa in pool_apply:
            yield pa.get(99)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), '  ', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), end=' ')
                sys.stdout.flush()
        # close()：禁止进程池再接收任务
	    # terminate()：强行终止进程池，不论是否有任务在执行
	    # join()：在close()或terminate()之后进行，等待进程退出
        pool.close()
        pool.join()
        del pool

    else:

        for i,t in tasks.items():
            yield call_func(t)
            completed_count += 1

            if verbose:
                print('\r', completed_count, '/', len(tasks), end=' ')
                sys.stdout.flush()
	
    if verbose:		print('tomominer.parallel.multiprocessing.util.run_iterator()', 'end', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
run_batch = run_iterator #alias



def call_func(t):

    if 'func' in t:
        assert 'module' not in t
        assert 'method' not in t
        func = t['func']
    else:
        modu = importlib.import_module(t['module'])
        # getattr() 函数用于返回一个对象属性值, 寻找modu里的t['method']
        func = getattr(modu, t['method'])
    # *展开list作为输入; 使用**将kwargs展开，变成了需要的关键字参数
    r = func(*t['args'], **t['kwargs'])
    return {'id':t['id'], 'result':r}



def random_rotation_matrix():
    m = np.random.random( (3,3) ) # 随机生成3*3矩阵
    u,s,v = np.linalg.svd(m) # 分解出正交阵，也就是旋转矩阵

    return u



def rotate3d_zyz(data, Inv_R, center=None, order=2):
    """Rotate a 3D data using ZYZ convention (phi: z1, the: x, psi: z2).
    """
    # Figure out the rotation center
    if center is None:
        cx = data.shape[0] / 2
        cy = data.shape[1] / 2
        cz = data.shape[2] / 2
    else:
        assert len(center) == 3
        (cx, cy, cz) = center

    
    from scipy import mgrid
    # 生成以center作为中心的三维坐标，并reshape为3*n的形状
    grid = mgrid[-cx:data.shape[0]-cx, -cy:data.shape[1]-cy, -cz:data.shape[2]-cz]
    temp = grid.reshape((3, np.int32(grid.size / 3)))
    temp = np.dot(Inv_R, temp)
    grid = np.reshape(temp, grid.shape)
    grid[0] += cx
    grid[1] += cy
    grid[2] += cz

    # Interpolation
    # reverse warping, so using the inv_R
    from scipy.ndimage import map_coordinates
    d = map_coordinates(data, grid, order=order, mode = 'reflect')

    return d


def smooth(v, sigma):
    assert  sigma > 0
    return SN.gaussian_filter(input=v, sigma=sigma)

    
    
def multiply(v, alpha, beta):
    return v * alpha + beta


    
def occlusion(v, start, end):    
    """
    挖空中心
    """
    v[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
    
    return v    



def augment(v, Inv_R, sigma, alpha, beta, start, end):

    v[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 0
    
    va = rotate3d_zyz(v, Inv_R)
            
    va[va == 0] = np.random.normal(loc=0.0, scale=1.0, size = np.sum(va == 0)) 
    
    vs = SN.gaussian_filter(input=va, sigma=sigma)
    
    if np.random.uniform() < 0.5:
        va = va + 0.25 * (vs - SN.gaussian_filter(input=va, sigma=5.0)) #LOG? Laplacian of Gaussian
        
    else:
        va = vs     
        
    va = va * alpha + beta      

    return va



def data_augmentation(x_train, factor = 2):
    '''
    rotation, smooth, 挖空
    '''
    if factor > 1:

        x_train_augmented = []
        
        image_size = x_train.shape[1]
        
        x_train_augmented.append(x_train)

        for f in range(1, factor):
            ts = {}        
            for i in range(len(x_train)): #batch size                      
                t = {}                                                
                t['func'] = augment                                   
                                                      
                # prepare keyword arguments                                                                                                               
                args_t = {}           
                args_t['v'] = x_train[i,:,:,:,0]
                args_t['Inv_R'] = random_rotation_matrix()
                args_t['sigma'] = np.random.uniform(0, 2.0) 
                args_t['alpha'] = np.random.uniform(0.8, 1.2)  
                args_t['beta'] = np.random.uniform(-0.2, 0.2) 
                start = np.random.randint(0, image_size, 3)
                args_t['start'] = start
                args_t['end'] = start + np.random.randint(0, image_size/4, 3)
   
                t['kwargs'] = args_t                                                  
                ts[i] = t                                                       
                                                                      
            rs = run_batch(ts, worker_num = Worker_Num) #并行运算生成batch的augenment的training data
            x_train_f = np.expand_dims(np.array([_['result'] for _ in rs]), -1)
            
            x_train_augmented.append(x_train_f)
            
        x_train_augmented = np.concatenate(x_train_augmented)
    
    else:
        x_train_augmented = x_train                        

    return x_train_augmented


def data_zoom(x_train, factor):
    '''
    并行运算处理图像缩放
    '''
    ts = {}        
    for i in range(len(x_train)):                       
        t = {}                                                
        t['func'] = rescale     # skimage的                              
                                              
        # prepare keyword arguments                                                                                                               
        args_t = {}           
        args_t['image'] = x_train[i,:,:,:,0]
        args_t['scale'] = (factor, factor, factor)                                                   
                                                                                                               
        t['kwargs'] = args_t                                                  
        ts[i] = t                                                       
                                                              
    rs = run_batch(ts, worker_num=Worker_Num)
    x_train_zoomed = np.expand_dims(np.array([_['result'] for _ in rs]), -1)            

    return x_train_zoomed



def one_hot(a, num_classes):
    
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])   



def smooth_labels(labels, factor=0.1):
	# smooth the labels
	labels *= (1 - factor)
	labels += (factor / labels.shape[1])
 
	# returned the smoothed labels
	return labels                                            



def remove_empty_cluster(labels):
    labels_unique = np.unique(labels)
    for i in range(len(np.unique(labels))):
        labels[labels == labels_unique[i]] = i

    return labels

                        
#not used
def merge_small_cluster(labels, labels_proba, n = 100):
    small_cluster = np.unique(labels)[np.array([np.sum(labels == i) for i in np.unique(labels)]) < n]

    labels_sort = np.argsort(-labels_proba) # label_proba 由大到小排列的index
    
    mask = np.isin(labels_sort, small_cluster) # 标记small cluster为True
    
    labels_sort_index = np.argmin(mask, 1) # 标记出除去small cluster后的最大的index
    
    labels_new = np.array([labels_sort[i, labels_sort_index[i]] for i in range(len(labels))])
    
    return labels_new




def statistical_fitting(features, labels, candidateKs, K, reg_covar, i):
    # features.shape = [15,1,1,1,32]
    # labels = None
    # candidateKs=[12], k = 12
    # K=None
    # reg_covar=0.01
    # i = 0

#    pca = PCA(n_components=32)  
#    features_pca = pca.fit_transform(features) 

    features_pca = features
    reg_covar = np.max([reg_covar * (0.5**i), 1e-6]) # 防止不可逆的对角线加和   
    
    labels_K = [] 
    models = []
    BICs = [] 
                                        
    for k in candidateKs: 
        if k == K: 
            try:
                weights_init = np.array([np.sum(labels == j)/np.float(len(labels)) for j in range(k)]) 
                means_init = np.array([np.mean(features_pca[labels == j], 0) for j in range(k)]) 
                precisions_init = np.array([np.linalg.inv(np.cov(features_pca[labels == j].T) + reg_covar * np.eye(features_pca.shape[1])) for j in range(k)]) 
 
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=5, n_init=1, random_state=i,  
                                        weights_init = weights_init, precisions_init = precisions_init,  means_init=means_init, init_params = 'kmeans') 
 
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca)

            except:     
                gmm_0 = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=reg_covar, max_iter=5, n_init=1, random_state=i, init_params = 'kmeans') 
                features_pca = np.squeeze(features_pca)
                gmm_0.fit(features_pca) 
                labels_k_0 = gmm_0.predict(features_pca) 
                         
         
            gmm_1 = GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, reg_covar=reg_covar, max_iter=1000, n_init=2, random_state=i, init_params = 'kmeans') 
            gmm_1.fit(features_pca) 
            labels_k_1 = gmm_1.predict(features_pca) 
            # 随机初始化与经验初始化的模型比较 
            m_select = np.argmin([-gmm_0.score(features_pca), -gmm_1.score(features_pca)])
            
            if m_select == 0: 
                labels_K.append(labels_k_0) 
                BICs.append(-gmm_0.score(features_pca))
                gmm = gmm_0 
                models.append(gmm)
             
            else: 
                labels_K.append(labels_k_1) 
                BICs.append(-gmm_1.score(features_pca))
                gmm = gmm_1 
                models.append(gmm)
         
        else: 
            gmm = GaussianMixture(n_components=k, covariance_type='full', tol=0.0001, reg_covar=reg_covar, max_iter=1000, n_init=2, random_state=i, init_params = 'kmeans') 
         
            features_pca = np.squeeze(features_pca) # features_pca.shape=(15,32)
            gmm.fit(features_pca) 
            labels_k = gmm.predict(features_pca) # label_k.shape=(15,)

            labels_K.append(labels_k) 
             
            BICs.append(-gmm.score(features_pca)) 
            models.append(gmm)
    
    labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])
    gmm = models[np.argmin(BICs)]
    labels_temp_proba = gmm.predict_proba(features_pca)  # labels_temp_proba.shape=(15,12)，表示第i个样本属于15个标签中某一个的概率                   
    # predict_proba返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
    K_temp = len(np.unique(labels_temp)) 
     
    if K_temp == K: 
        same_K = True 
    else: 
        same_K = False 
        K = K_temp     

    print('Estimated K:', K)
    
    return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm         
    # labels_temp.shape=(15,)
    # labels_temp_proba.shape=(15,12)
    # K = None
    # same_K = False
    # feature_pca = (15,32)


def normalize_data(train_data):
    val_min = np.min(train_data)
    val_max = np.max(train_data)
    train_data = (train_data - val_min)/(val_max - val_min)

    return train_data

def prepare_training_data(x_train, labels_temp_proba, labels, n):
        # x_train_origin.shape=(15,24,24,24,1)
        # label_temp_proba.shape=(15,12)
        # labels.shape=(15,)
    
        # n代表着重复生成data的次数
        #    label_one_hot = labels_temp_proba


    len_unique = len(np.unique(labels))
    label_one_hot = one_hot(labels, len_unique)

    index = np.array(range(x_train.shape[0] * n))
    # 函数的语法是np.tile(a, reps)，a表示类数组元素（不仅可以是ndarray数组，也可以是列表、元组等），reps用来定义各个方向上的拷贝数量。reps参数可以记忆成repeat shape，也即拷贝性扩展的形状。
    labels_tile = np.tile(labels, n) # m*n
    # labels_tile.shape=(20,)
    labels_proba_tile = np.tile(labels_temp_proba, (n, 1)) # (m*n)*n m*n的组合在列重复n次
    # labels_proba_tile.shape=(20,3)
    labels_np = []
    
    for i in range(len(np.unique(labels))):
        npi = np.maximum(0, 0.5 - labels_proba_tile[:, i][labels_tile != i]) # 逐元素比较， 非 i th cluster的样本第i个cluster的概率
        labels_np.append(npi/np.sum(npi))
    # n =1
    x_train_augmented = data_augmentation(x_train, n)   # x_train_augmented.shape=(15,24,24,24,1)
    
    _pos = data_augmentation(x_train, n + 1)    # _pos.shape=(30,24,24,24,1)
    x_train_augmented_pos = _pos[x_train.shape[0]:]# x_train_augmented_pos.shape=(15,24,24,24,1)
    
    # labels_tile.shape=[20,] , [2,1,2,1,,,,]
    #index 抽样

    index_negative = np.array([np.random.choice(a = index[labels_tile != labels_tile[i]], p = labels_np[labels_tile[i]]) for i in range(len(index))])   # index_negative=(15,)

    x_train_augmented_neg = data_augmentation(x_train, n + 1)[x_train.shape[0]:][index_negative]
    np.random.shuffle(index)              
    
    x_train_permute = [x_train_augmented[index].copy(), x_train_augmented_pos[index].copy(), x_train_augmented_neg[index_negative].copy()] 
    labels_permute = [np.tile(label_one_hot, (n, 1))[index].copy(), np.tile(label_one_hot, (n, 1))[index].copy(),np.tile(label_one_hot, (n, 1))[index_negative].copy()] 

    return label_one_hot, x_train_permute, labels_permute


# 网络模型
class YOLO_Model(nn.Module):
    def __init__(self, hidden_channel=32):
        super(YOLO_Model, self).__init__()

        self.hidden_channel = hidden_channel
        self.image_size = 24

        self.conv1 = nn.Conv3d(1, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=self.image_size-2, stride=1)
        self.bn = nn.BatchNorm3d(self.hidden_channel)


        self.conv1 = nn.Conv3d(1, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))

        self.conv2 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv3 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))

        self.conv4 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn4 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))

        self.conv5 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn5 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv6 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn6 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv7 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn7 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv8 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn8 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv9 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn9 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        self.conv10 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=1, padding=1)
        self.bn10 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel),
            nn.MaxPool3d(kernel_size=self.image_size, stride=1))


        # self.fc = nn.Linear(in_features=1236, out_features=32)
        # self.fc = nn.Linear(in_features=self.hidden_channel*36+84, out_features=32)
        self.fc = nn.Linear(in_features=self.hidden_channel*10, out_features=32)
        #########################################

        self.conv1_1 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=2, padding=1)
        self.conv1_2 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=2, padding=1)
        self.conv1_3 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=2, padding=1)
        self.conv1_4 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=2, padding=1)
        self.conv1_5 = nn.Conv3d(self.hidden_channel, self.hidden_channel, (3, 3, 3), stride=2, padding=1)
        
        
        self.bn1_1 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel))

        self.bn1_2 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel))

        self.bn1_3 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel))
        
        self.bn1_4 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel))

        self.bn1_5 = nn.Sequential( 
            nn.ReLU(),
            nn.BatchNorm3d(self.hidden_channel))
        
    def forward(self,input):    # input=[3,15,1,24,24,24]

        # x1 = input[0]   # x1=[15,1,24,24,24]
        # x2 = input[1]   # x2=[15,1,24,24,24]
        # x3 = input[2]   # x3=[15,1,24,24,24]

        x = input
        x = self.conv1(x)
        x = self.bn1_1(x)    # m1.shape=[15,32,3,3,3]

        x = self.conv1_1(x)
        x = self.bn1_2(x)
        
        x = self.conv1_2(x)
        x = self.bn1_3(x)
        x = self.conv1_3(x)
        x = self.bn1_4(x)
        x = self.conv1_4(x)
        x = self.bn1_5(x)
        x = self.conv1_5(x)


        x = x.view(-1, x.size()[1])
        # x_concat = torch.cat((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10), 1)

        # x_concat = x_concat.view(-1, x_concat.size()[1])
        
        # x_out = self.fc(x_concat)

        return x



class YOPO_classification(nn.Module):
    def __init__(self, num_labels=12,vector_size=32):
        super(YOPO_classification, self ).__init__()

        self.fc = nn.Linear(in_features=vector_size, out_features=num_labels)

        self.soft = nn.Softmax()
        # self.soft = nn.Sigmoid()
        # self.soft = nn.ReLU()

    def forward(self,input):    # input=[44,24,24,24,1]

        x = input
        x = self.fc(x)
        x = self.soft(x)

        return x


class CombinedModel(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        self.submodel1 = submodel1
        self.submodel2 = submodel2
 
    def forward(self, x):
        x = self.submodel1(x)
        x = self.submodel2(x)
        return x
 
##############################################################
##############################################################
##############################################################
def convergence_check(i, M, labels_temp, labels, done):
    if i > 0:
        if np.sum(labels_temp == labels)/np.float(len(labels)) > 0.999: 
            done = True 

    i += 1 
    if i == M: 
        done = True
    labels = labels_temp
    
    return i, labels, done                 



import torch.utils.data as Data


if __name__ == '__main__':
    np.random.seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

    ### Define Parameters Here ###
    gt = np.repeat(range(10), 1000)  
    image_size = 24   ### subtomogram size ### 
    candidateKs = [12]   ### candidate number of clusters to test  
    BATCH_SIZE = 64      
    M = 40   ### number of iterations ### 
    lr = 0.0001   ### CNN learning rate ### 

    label_smoothing_factor = 0.2   ### label smoothing factor ### 
    reg_covar = 0.01 
    ### path for saving keras model, should be a .h5 file ### 
    model_path = 'E:\Code\CryoET\picking\picking\data_emd4603\synechocystis1.h5'
    ### path for saving labels, should be a .pickle file ###
    label_path = 'E:\Code\CryoET\picking\picking\data_emd4603\labels_1.pickle'
 
    ### Load Dataset Here ###     
    import h5py         
    h5f = h5py.File('E:\Code\CryoET\picking\picking\data_emd4603\data.h5','r')                                                           
    x_train = h5f['dataset_1'][:] # only 'dataset_1', x_train.shape=(16265, 24, 24, 24, 1)
    h5f.close() 

    x_train_origin = x_train[:15]

    ### Generalized EM Process ### 
    K = None 
    labels = None
    DBI_best = np.inf

    done = False 
    i = 0


    yolo_model = YOLO_Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo_model.to(device)
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(yolo_model.parameters(),lr = 0.01,momentum=0.5)

    x_train = [x_train_origin, x_train_origin, x_train_origin]   # x_train=[3,15,24,24,24,1]
    x_train = torch.tensor(x_train)
    torch_dataset = Data.TensorDataset(x_train, x_train)  # 对给定的 tensor 数据，将他们包装成 dataset

    train_dataloader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,       # torch TensorDataset format
        batch_size=BATCH_SIZE,       # mini batch size
        shuffle=True,                # 要不要打乱数据 (打乱比较好)
        num_workers=1,               # 多线程来读数据
    )

    # 训练
    epoch_list = []

    for epoch in range(1):
        running_loss = 0.0
        epoch_list.append(epoch+1)
        # for epoch in range(10):
        for i, data in enumerate(train_dataloader, 0):
            input, target = data                  # input =[3,15,24,24,24,1]
            input = input.permute(0,1,5,2,3,4)    # input =[3,15,1,24,24,24]
            input, target = input.to(device),target.to(device)

            input1 = input[0]   # x1=[15,1,24,24,24]
            input2 = input[1]   # x2=[15,1,24,24,24]
            input3 = input[2]   # x3=[15,1,24,24,24]

            features1 = yolo_model(input1)
            features = features1.unsqueeze(dim=1)
            features = features.unsqueeze(dim=1)
            features = features.unsqueeze(dim=1)
            # features=torch.size([15,1,1,1,32])
            features = features.detach().numpy()
            ### Feature Clustering ###                                              
            labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = statistical_fitting(features = features, labels = labels, candidateKs = candidateKs,K = K, reg_covar = reg_covar, i = i) 

            # labels_temp.shape=(15,)
            # labels_temp_proba.shape=(15,12)
            # K = None
            # same_K = False
            # feature_pca = (15,32)


            ### Matching Clusters by Hungarian Algorithm ### 
            if same_K: 
                labels_temp, col_ind = align_cluster_index(labels, labels_temp)
                labels_temp_proba = labels_temp_proba[:,col_ind]

            i, labels, done = convergence_check(i = i, M = M, labels_temp = labels_temp, labels = labels, done = done) # i is added 1 here
                
            print('Cluster sizes:', [np.sum(labels == k) for  k in range(K)])         


    ### Validate Clustering by distortion-based DBI ###             
        DBI = DDBI(features_pca, labels) 
 
        if DBI < DBI_best: 
            if i > 1:

                torch.save(yolo_model.state_dict(), model_path) # .pt or .pth
               
                labels_best = labels.copy()   ### save current labels if DDBI improves ###  
                 
                pickle_dump(labels_best, label_path) 

            DBI_best = DBI                   
        print('DDBI:', DBI, '############################################') 


        ### Permute Samples ###             

        label_one_hot, x_train_permute, labels_permute = prepare_training_data(x_train = x_train_origin, labels_temp_proba = labels_temp_proba, labels = labels, n = 1)
        # input
        # x_train_origin.shape=(15,24,24,24,1)
        # label_temp_proba.shape=(15,12)
        # labels.shape=(15,)

        # output
        # label_one_hot.shape=(15,12)
        # x_train_permute.shape=(15,24,24,24,1)*3
        # labels_permute.shape=(15,12)*3

         
    ### Finetune new model with current estimated K ### 
        if not same_K:
            print('finished.........................')                         
            model_classification = YOPO_classification(num_labels=K, vector_size = 32)
            # K = 12
            # criterion = nn.CrossEntropyLoss(reduction='mean')
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.MSELoss()

            optimizer = optim.Adam(model_classification.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

            features = np.squeeze(features)

            _features = [features, features, features] # _features 有三个向量，每个都是(15,32)
            _label = [label_one_hot, label_one_hot, label_one_hot]
            # _label = [label_one_hot, label_one_hot, label_one_hot, np.zeros(features.shape[0]), np.zeros(features.shape[0])]
            # # _label有5个向量，前三个是(15,12)，后两个是(15,)

            _features = torch.tensor(_features)
            _label = torch.tensor(_label,dtype=torch.float32)
            class_dataset = Data.TensorDataset(_features, _label)  

            classification_dataloader = Data.DataLoader(
                # 从数据库中每次抽出batch size个样本
                dataset=class_dataset,       
                batch_size=BATCH_SIZE,       
                shuffle=True,                
                num_workers=1,               
            )

            # 训练
            # model_classification.train()
            epoch_list = []
            for classification_epoch in range(10):
                running_loss = 0.0
                epoch_list.append(classification_epoch+1)
                for i, data in enumerate(classification_dataloader, 0):
                    input, target = data    # input.shape=[3,15,32], target.shape=[3,15,12]
                    input, target = input.to(device),target.to(device)
                    y_pred = model_classification(input)
                    loss = criterion(y_pred, target) # y_pred=float32,target=float64
                    optimizer.zero_grad()
                    loss.backward()
                    # loss.backward(torch.ones_like(loss))
                    optimizer.step()

                    running_loss += loss.item()

            ### New YOPO ### 
            # parallel_model = YOLO_Model()
            submodel1 = YOLO_Model()
            submodel2 = YOPO_classification()
            parallel_model = CombinedModel(submodel1,submodel2)


        ### CNN Training ### 
        lr *= 0.95 
        # criterion = nn.CrossEntropyLoss(reduction='mean')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model_classification.parameters(),lr = 0.01,momentum=0.5)          

        _label2 = [labels_permute[0], labels_permute[1], labels_permute[2]]
        # x_train_permute = [3,15,24,24,24,1]
        x_train_permute = torch.tensor(x_train_permute)
        _label2 = torch.tensor(_label2,dtype=torch.float32)     # _label2=[3,15,12]
        parallel_dataset = Data.TensorDataset(x_train_permute, _label2)  

        parallel_dataloader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=parallel_dataset,       
            batch_size=BATCH_SIZE,       
            shuffle=True,                
            num_workers=1,               
        )

        for parallel_epoch in range(1):
            for i, data in enumerate(parallel_dataloader, 0):
                input, target = data
                input = input.permute(0,1,5,2,3,4)    # input =[3,15,1,24,24,24]
                input, target = input.to(device),target.to(device)
                
                input1 = input[0]
                input2 = input[1]
                input3 = input[2]
                
                y_pred1 = parallel_model(input1).unsqueeze(0)    # y_pred1=[15,32]
                y_pred2 = parallel_model(input2).unsqueeze(0)    # y_pred2=[15,32]
                y_pred3 = parallel_model(input3).unsqueeze(0)    # y_pred3=[15,32]
                y_pred = torch.cat([y_pred1,y_pred2,y_pred3], 0) # y_pred=[3,15,12], float32
                # y_pred = torch.tensor[y_pred1,y_pred2,y_pred3]
                loss = criterion(y_pred, target)    # target=[3,15,12],float64
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

