import time, pickle
from tqdm import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Conv3D, Activation, GlobalAveragePooling3D, Dropout, \
    BatchNormalization, Concatenate, ELU, GaussianDropout, GlobalMaxPooling3D, MaxPooling3D, Subtract, LeakyReLU

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
from GMMU.gmmu_cavi_stable_new import CAVI_GMMU as GMM
from deepdpm_gmmu.split_merge_function_new import *                               

import scipy.ndimage as SN
import gc

#from gmm_tf import GaussianMixture as GMM 

Worker_Num = 48


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
    if np.isnan(DBI):
        print(np.unique(labels)) 
        print(means_init)
        print(precisions_init)
    return DBI 


# def DDBI_tf(features, labels):
#     """
#     Davies Bouldin index with different definition of the compact and distance between clusters
#     """
#     features = tf.convert_to_tensor(features)
#     labels = tf.convert_to_tensor(labels)

#     unique_labels = tf.unique(labels)[0]
#     num_clusters = len(unique_labels)

#     means_init = tf.convert_to_tensor([tf.reduce_mean(tf.boolean_mask(features, labels == i), axis=0) for i in unique_labels])
#     precisions_init = tf.convert_to_tensor([tf.linalg.inv( tfp.stats.covariance(tf.boolean_mask(features, labels == i), sample_axis=0, event_axis=1) \
#                                                            + 1e-6 * tf.eye(features.shape[1])) for i in unique_labels])

#     T = tf.convert_to_tensor([tf.reduce_mean(tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(features[labels == i] - means_init[i], precisions_init[i]), \
#                                                                       tf.transpose(features[labels == i] - means_init[i])))) for i in unique_labels])

#     D = tf.convert_to_tensor([tf.linalg.diag_part(tf.linalg.matmul(tf.linalg.matmul(means_init - means_init[i], precisions_init[i]), \
#                                                        tf.transpose(means_init - means_init[i]))) for i in unique_labels])

#     DBI_matrix = np.zeros((num_clusters, num_clusters))

#     for i in range(num_clusters):
#         for j in range(num_clusters):
#             if i != j:
#                 DBI_matrix[i, j] = (T[i] + T[j]) / (D[i, j] + D[j, i])

#     DBI = tf.reduce_mean(tf.reduce_max(DBI_matrix, axis=0))

#     return DBI.numpy()                 




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
                weights_init = np.array([np.sum(labels == j)/(len(labels)) for j in range(k)]) 
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
            
            # print(m_select) 
             
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
         
            
            features_pca = np.squeeze(features_pca)
            gmm.fit(features_pca) 
            labels_k = gmm.predict(features_pca) 

            labels_K.append(labels_k) 
             
            BICs.append(-gmm.score(features_pca)) 
            models.append(gmm)
    
    labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])
    gmm = models[np.argmin(BICs)]
    labels_temp_proba = gmm.predict_proba(features_pca)                     
     
    K_temp = len(np.unique(labels_temp)) 
     
    if K_temp == K: 
        same_K = True 
    else: 
        same_K = False 
        K = K_temp     

    print('Estimated K:', K)
    
    return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm         
         


# def statistical_fitting_tf(features, labels, candidateKs, K, reg_covar, i, u_filter = True):
#     '''
#     introduce the tensoeflow version GMM
#     '''
# #    pca = PCA(n_components=32)  
# #    features_pca = pca.fit_transform(features) 

#     features_pca = features

#     reg_covar = np.max([reg_covar * (0.5**i), 1e-6])    
    
#     labels_K = [] 
#     models = []
#     BICs = [] 
                                        
#     for k in candidateKs: 
#         if k == K: 
#             gmm_0 = GMM(n_cluster = k, a_o = None, b_o = None, u_filter = u_filter) 
#             gmm_0.fit(features_pca) 
#             labels_k_0 = gmm_0.hard_assignment()
                                  
#             gmm_1 = GMM(n_cluster = k, a_o = None, b_o = None, u_filter = u_filter) 
#             gmm_1.fit(features_pca) 
#             labels_k_1 = gmm_1.hard_assignment()
             
#             m_select = np.argmin([gmm_0.bic(features_pca).numpy(), gmm_1.bic(features_pca).numpy()])
            
#             # print(m_select) 
             
#             if m_select == 0: 
#                 labels_K.append(labels_k_0) 
                 
#                 BICs.append(gmm_0.bic(features_pca).numpy())
                
#                 gmm = gmm_0 

#                 models.append(gmm)
             
#             else: 
#                 labels_K.append(labels_k_1) 
                 
#                 BICs.append(gmm_1.bic(features_pca).numpy())
                
#                 gmm = gmm_1

#                 models.append(gmm) 
         
#         else: 
#             gmm = GMM(n_cluster = k, a_o = None, b_o = None, u_filter = u_filter)
         
#             gmm.fit(features_pca) 
#             labels_k = gmm.hard_assignment()

#             labels_K.append(labels_k) 
             
#             BICs.append(gmm.bic(features_pca).numpy())

#             models.append(gmm) 
    
#     labels_temp = remove_empty_cluster(labels_K[np.argmin(BICs)])
#     gmm = models[np.argmin(BICs)]
#     labels_temp_proba = gmm.soft_assignment().numpy()                     
     
#     K_temp = len(np.unique(labels_temp)) 
     
#     if K_temp == K: 
#         same_K = True 
#     else: 
#         same_K = False 
#         K = K_temp     

#     print('Estimated K:', K)
    
#     return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm         


def distance_matrix(t, n_components, n_feature):
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


def statistical_fitting_tf_split_merge(features, labels, candidateKs, K, reg_covar, it, u_filter = True,\
                                        u_filter_rate=0.0025, alpha = 1.0):
    '''
    introduce the tensoeflow version GMM
    '''
#    pca = PCA(n_components=32)  
#    features_pca = pca.fit_transform(features) 

    features_pca = features
    n_features = features_pca.shape[1]

    reg_covar = np.max([reg_covar * (0.5**it), 1e-6])    
    
    labels_K = [] 
    models = []
    BICs = [] 
    k_new = K
    k_original = K


    if K in candidateKs:
        gmm_0 = GMM(n_cluster = K, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate, \
                    reg_covar=reg_covar)
        gmm_0.fit(features_pca) 
        labels_k_0 = gmm_0.hard_assignment()
        alpha_o, beta_o, m_o, w_o, nu_o, b_o, a_o = gmm_0.parameters_o()
        r_cluster = gmm_0.soft_assignment()
        hardcluster_label = gmm_0.hard_assignment()

        #lambda_pi, lambda_beta, lambda_m, lambda_w, lambda_nu, b_o, a_o= gmm_0.parameters()
        #print(lambda_w,tf.linalg.det(lambda_w))

        #split
        for i in range(1, K+1): #ignore uniform
            x_i = features_pca[hardcluster_label == i,:]
            if x_i.shape[0] <= 1 :
                    continue
            gmm_i = GaussianMixture(n_components=2, covariance_type='full', \
                                    tol=0.001, random_state=i, init_params = 'kmeans', reg_covar=reg_covar)
            gmm_i.fit(x_i) 
            subcluster_i_hard_label = gmm_i.predict(x_i)
            x_i1 = x_i[subcluster_i_hard_label==0, :]
            x_i2 = x_i[subcluster_i_hard_label==1, :]
            if x_i1.shape[0] == 0 or x_i2.shape[0] == 0:
                    continue
            r_all = r_cluster[:, i]  # (n, )
            r_sub = tf.expand_dims(r_all, axis=-1) * tf.constant(gmm_i.predict_proba(features_pca))  # (n, 2)
            r_c1 = r_sub[:, 0]  # (n, )
            r_c2 = r_sub[:, 1]  # (n, )

            hs = acceptpro_split_hs(features_pca, x_i, x_i1, x_i2, r_all, r_c1, r_c2, \
                                    m_o, beta_o, w_o, nu_o, alpha)
            #if not np.isnan(hs):
            #    print('hs', hs)
            if tf.random.uniform(shape=[], minval=0., maxval=1) < tf.cast(hs, tf.float32):
                if k_new >= np.max(candidateKs):
                    break
                else:
                    k_new = k_new + 1
        
        if k_new != K:
            gmm_1 = GMM(n_cluster = k_new, a_o = None, b_o = None, u_filter = u_filter, \
                        u_filter_rate = u_filter_rate, reg_covar=reg_covar)
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
            merge_pair = tf.argsort(dm)[:, 1: (n_pair + 1)]
            for i, pairs in enumerate(merge_pair):
                if (i + 1) in list(np.array(merged_list).flat) or (i + 1) in nan_cluster_list:
                    # plus 1 because considering uniform pi_0
                    continue
                if k_new2 <= np.min(candidateKs):
                    break
                mask_i = hardcluster_label == (i + 1)  # plus 1 because considering uniform pi_0
                X_c1 = tf.boolean_mask(features_pca, mask=mask_i)
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
                    X_c2 = tf.boolean_mask(features_pca, mask=mask_j)
                    X_merge = tf.concat([X_c1, X_c2], 0)

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
                            (tf.random.uniform(shape=[], minval=0., maxval=1) < tf.cast(hm, tf.float32)):
                        merged_list.append([i + 1, pair + 1])
                        if k_new2 <= np.min(candidateKs):
                            break
                        else:
                            k_new2 = k_new2 - 1
        
        if k_new2 != k_new:
            gmm = GMM(n_cluster = k_new2, a_o = None, b_o = None, u_filter = u_filter, \
                       u_filter_rate = u_filter_rate, reg_covar=reg_covar)
            gmm.fit(features_pca)
        else:
            gmm = gmm_1
        
        labels_k = gmm.hard_assignment()
        K_temp = k_new2 #ignore uniform 
        #print('hm', hm)
        #print('hs', hs)
                                        
    
    else: 
        K_temp = int(0.5*(np.min(candidateKs) + np.max(candidateKs)))   # K_temp=13
        gmm = GMM(n_cluster = K_temp, a_o = None, b_o = None, u_filter = u_filter, u_filter_rate = u_filter_rate,\
                  reg_covar=reg_covar) 
        
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
            means_init = np.array([np.mean(features_pca[labels == j], 0) for j in range(1,K+1)])
            precisions_init = np.array(
                [np.linalg.pinv(np.cov(features_pca[labels == j].T)) for j in range(1,K+1)])
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
    labels_temp_proba = gmm.soft_assignment().numpy()
    # if k != the number of Gaussian cluster predicted, it means there is some clusters disappearing, 
    # so ignore this iteration
    
    # unique_temp = (len(np.unique(labels_temp))-1)
    # if K != unique_temp:
    #     same_K = False
    #     labels_temp = labels

    print('Estimated K:', K)
    #if np.any(np.isnan(labels_temp_proba)):
    #    print('###in###there is nan in labels_temp_proba')
    #    print(labels_temp_proba)
    return labels_temp_proba, labels_temp, K, same_K, features_pca, gmm   


def nmse(y_true, y_pred):
    return tf.maximum(0., 2 - tf.reduce_mean(tf.square(y_pred), axis = -1))



def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred), axis = -1)



from tensorflow.keras.layers import Layer


class CosineSimilarity(Layer):

    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return (None, 1)


    def call(self, tensors):
        f_A, f_B = tensors
        output = self._cosinesimilarity(f_A = f_A, f_B = f_B)
        return output

        
    def get_config(self):
        base_config = super(CosineSimilarity, self).get_config()
        
        return dict(list(base_config.items()))


    def _cosinesimilarity(self, f_A, f_B):
        denominator = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(f_A), axis = -1)), tf.sqrt(tf.reduce_sum(tf.square(f_B), axis = -1)))
        
        nominator = tf.reduce_sum(tf.multiply(f_A, f_B), axis = -1)
        
        return nominator/denominator



def NSNN(y_true, y_pred):
    return tf.math.log(tf.reduce_sum(tf.math.exp(tf.abs(y_pred)/0.2)))/64.



def SNN(y_true, y_pred):
    return -tf.math.log(tf.reduce_sum(tf.math.exp(y_pred/0.2)))/64.


#def NSNN(y_true, y_pred):
#    return 0.5 * tf.abs(y_pred)



#def SNN(y_true, y_pred):
#    return 1 - y_pred



def update_output_layer(K, label_one_hot, batch_size, model_feature, features, lr):

    model_classification = YOPO_classification(num_labels=K, vector_size = 32) #YOPO是一个分类网络

    optimizer = tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 
     
    model_classification.compile(optimizer= optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', SNN, NSNN], loss_weights = [1, 0, 0, 0, 0]) 
         
    model_classification.fit([features, features, features], \
            [label_one_hot, label_one_hot, label_one_hot, np.zeros(features.shape[0]), np.zeros(features.shape[0])], epochs=10, batch_size=batch_size, shuffle=True, verbose = 0) 
     
    ### New YOPO ### 
    model = tf.keras.Model(model_feature.input, model_classification(model_feature.output)) 
#    optimizer = tf.keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

#    model.compile(optimizer= optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', SNN, NSNN])

    return model



def prepare_training_data(x_train, labels_temp_proba, labels, n):
    # n代表着重复生成data的次数
#    label_one_hot = labels_temp_proba
    
    label_one_hot = one_hot(labels, len(np.unique(labels))) 
     
    index = np.array(range(x_train.shape[0] * n))
    
    labels_tile = np.tile(labels, n) # m*n
    
    labels_proba_tile = np.tile(labels_temp_proba, (n, 1)) # (m*n)*n m*n的组合在列重复n次
    
    labels_np = []
    
    for i in range(len(np.unique(labels))):
        npi = np.maximum(0, 0.5 - labels_proba_tile[:, i][labels_tile != i]) # 逐元素比较， 非 i th cluster的样本第i个cluster的概率
        
        labels_np.append(npi/np.sum(npi))
     
    x_train_augmented = data_augmentation(x_train, n)
    
    x_train_augmented_pos = data_augmentation(x_train, n + 1)[x_train.shape[0]:]
    
    #index 抽样
    index_negative = np.array([np.random.choice(a = index[labels_tile != labels_tile[i]], p = labels_np[labels_tile[i]]) for i in range(len(index))])
    
    x_train_augmented_neg = data_augmentation(x_train, n + 1)[x_train.shape[0]:][index_negative]

    np.random.shuffle(index)              
     
    x_train_permute = [x_train_augmented[index].copy(), x_train_augmented_pos[index].copy(), x_train_augmented_neg[index].copy()] 
          
    labels_permute = [np.tile(label_one_hot, (n, 1))[index].copy(), np.tile(label_one_hot, (n, 1))[index].copy(), np.tile(label_one_hot, (n, 1))[index_negative][index].copy()] 

    return label_one_hot, x_train_permute, labels_permute


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
    


def conv_block(img_shape = [24,24,24]):        
    kernel_initializer = tf.keras.initializers.orthogonal()         
    bias_initializer = tf.keras.initializers.zeros()
    input_shape = (img_shape[0], img_shape[1], img_shape[2], 1)  
                                                                 
    channel_axis = -1                                            
                                                                 
                                            
    main_input = Input(shape= input_shape, name='main_input')                                                       
                                                  
                                                  
                                   
    model = tf.keras.Model(inputs=main_input, outputs=x)                                                        
                                 
    return model
    
    
    

def feature_extraction(image_size):     # image_size = 24                   
    kernel_initializer = tf.keras.initializers.orthogonal()                                                   
    bias_initializer = tf.keras.initializers.zeros()                                                     
                                                  
    num_channels=1                  
    main_input = Input(shape = (None, None, None, num_channels))           

    input = GaussianDropout(0.5)(main_input)          

    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(input)

    # keras.layers.Conv3D(filters, 
    #                     kernel_size, 
    #                     strides=(1, 1, 1), 
    #                     padding='valid', 
    #                     data_format=None, 
    #                     dilation_rate=(1, 1, 1), 
    #                     activation=None)

    # filters: 卷积中滤波器的数量，表示输出张量的通道数
    # kernel_size: 卷积窗口的宽度和高度
    # strides: 卷积沿宽度和高度方向的步长
    # padding: 'VALID' 或 'SAME'，'VALID'表示无填充，'SAME'表示用0填充
    # data_format: 表示输入张量的维度顺序，默认为 [batch, dim1, dim2, dim3, channel], 如对分辨率为720×720视频(假设为连续5帧彩色图像)进行卷积，batch_size设为64，则输入向量的维度为 [64, 5, 720, 720, 3]
    # dilation_rate: 膨胀卷积的膨胀率
    # activation: 要使用的激活函数 



    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m1 = MaxPooling3D(pool_size = image_size-2, strides=1)(x)                  

    ###################################################
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m2 = MaxPooling3D(pool_size = image_size-4, strides=1)(x)    

    ###################################################
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m3 = MaxPooling3D(pool_size = image_size-6, strides=1)(x)                  

    ###################################################                       
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m4 = MaxPooling3D(pool_size = image_size-8, strides=1)(x)    
                                                  
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m5 = MaxPooling3D(pool_size = image_size-10, strides=1)(x)                  
                                            
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m6 = MaxPooling3D(pool_size = image_size-12, strides=1)(x)                                                      
                                            
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                        
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m7 = MaxPooling3D(pool_size = image_size-14, strides=1)(x)         
                                                  
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                        
    x = ELU()(x)                
    x = BatchNormalization()(x)                   
    m8 = MaxPooling3D(pool_size = image_size-16, strides=1)(x)         
                                                                                    
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                             
    x = ELU()(x)                         
    x = BatchNormalization()(x)          
    m9 = MaxPooling3D(pool_size = image_size-18, strides=1)(x)                  
                                            
    x = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m10 = MaxPooling3D(pool_size = image_size-20, strides=1)(x)


    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(input)
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m11 = MaxPooling3D(pool_size = image_size-3, strides=1)(x)                  
                                            
    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m12 = MaxPooling3D(pool_size = image_size-6, strides=1)(x)    

    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m13 = MaxPooling3D(pool_size = image_size-9, strides=1)(x)                  
                                            
    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m14 = MaxPooling3D(pool_size = image_size-12, strides=1)(x)    
                                                  
    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m15 = MaxPooling3D(pool_size = image_size-15, strides=1)(x)                  
                                            
    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m16 = MaxPooling3D(pool_size = image_size-18, strides=1)(x)                                                      
                                            
    x = Conv3D(44, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                        
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m17 = MaxPooling3D(pool_size = image_size-21, strides=1)(x)         
                                                  

    x = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(input)
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m18 = MaxPooling3D(pool_size = image_size-4, strides=1)(x)                  
                                            
    x = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m19 = MaxPooling3D(pool_size = image_size-8, strides=1)(x)    

    x = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m20 = MaxPooling3D(pool_size = image_size-12, strides=1)(x)                  
                                            
    x = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m21 = MaxPooling3D(pool_size = image_size-16, strides=1)(x)    
                                                  
    x = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m22 = MaxPooling3D(pool_size = image_size-20, strides=1)(x)                  
                                            

    x = Conv3D(96, (4, 4, 4), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(input)
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m23 = MaxPooling3D(pool_size = image_size-6, strides=1)(x)                  
                                            
    x = Conv3D(96, (4, 4, 4), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                
    x = ELU()(x)                         
    x = BatchNormalization()(x)             
    m24 = MaxPooling3D(pool_size = image_size-12, strides=1)(x)    

    x = Conv3D(96, (4, 4, 4), dilation_rate=(2, 2, 2), padding='valid', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                      
    x = ELU()(x)                         
    x = BatchNormalization()(x)                   
    m25 = MaxPooling3D(pool_size = image_size-18, strides=1)(x)                  
    
                
    x = Concatenate()([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25])  
                                         
    m = Dense(32, name='fc2', kernel_initializer=kernel_initializer, bias_initializer = bias_initializer)(x)                                     
                                                  
    mod = tf.keras.Model(inputs=main_input, outputs=m)                                                       
                                                  
    return mod



                                                                                                          
def YOPO_feature(image_size):
    kernel_initializer = tf.keras.initializers.orthogonal()
    bias_initializer = tf.keras.initializers.zeros()


    input_shape = (image_size, image_size, image_size, 1)       
                                      
    feature_extractor = feature_extraction(image_size) # this is a model

    main_input = Input(shape= input_shape, name='main_input')
    
    f1 = feature_extractor(main_input)
    
    positive_input = Input(shape= input_shape, name='pos_input')
    
    f2 = feature_extractor(positive_input)                                                                                 

    negative_input = Input(shape= input_shape, name='neg_input')
    
    f3 = feature_extractor(negative_input)                                                                                 

    mod = tf.keras.Model(inputs=[main_input, positive_input, negative_input], outputs=[f1, f2, f3])                                     
                                                                               
    return mod                                                                 
                                                                 


def YOPO_classification_old(num_labels, vector_size = 1024):

    input_shape = (vector_size,)
    
    dense = Dense(num_labels, activation='softmax')       
                                      
    main_input = Input(shape= input_shape, name='main_input')

    positive_input = Input(shape= input_shape, name='pos_input')

    negative_input = Input(shape= input_shape, name='neg_input')
        
    m1 = dense(main_input)
    
    m2 = dense(positive_input)
    
    m3 = dense(negative_input)
    
    s1 = CosineSimilarity()([main_input, positive_input])
    
    s2 = CosineSimilarity()([main_input, negative_input])

    mod = tf.keras.Model(inputs=[main_input, positive_input, negative_input], outputs=[m1, m2, m3, s1, s2])                                     
                                                                               
    return mod

# num_labels=12, vector_size=32
def YOPO_classification(num_labels, vector_size=1024):
    input_shape = (None, None, None, vector_size,)

    dense = Dense(num_labels, activation='softmax')
    main_input = Input(shape=input_shape, name='main_input')
    positive_input = Input(shape=input_shape, name='pos_input')
    negative_input = Input(shape=input_shape, name='neg_input')

    m1 = dense(main_input)      # m1=[None,None,None,None,12]

    m2 = dense(positive_input)  # m2=[None,None,None,None,12]

    m3 = dense(negative_input)  # m3=[None,None,None,None,12]

    s1 = CosineSimilarity()([main_input, positive_input])

    s2 = CosineSimilarity()([main_input, negative_input])

    mod = tf.keras.Model(inputs=[main_input, positive_input, negative_input], outputs=[m1, m2, m3, s1, s2])

    return mod



def convergence_check(i, M, labels_temp, labels, done):

    if i > 0:
        if np.sum(labels_temp == labels)/(len(labels)) > 0.999: 
            done = True 

    i += 1 
    if i == M: 
        done = True
        
    labels = labels_temp
    
    return i, labels, done                 


    

if __name__ == '__main__':
    np.random.seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

    ### Define Parameters Here ###
    gt = np.repeat(range(10), 1000)  
    
    image_size = 24   ### subtomogram size ### 
    candidateKs = [12]   ### candidate number of clusters to test  
          
    batch_size = 64      
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


    x_train = x_train[:15]

    # x_train = data_zoom(x_train, 24/32.) #Data 是(16265, 24, 24, 24, 1)，不是32的



    ### Generalized EM Process ### 
    K = None 
    labels = None
    DBI_best = np.inf

    done = False 
    i = 0
    
    strategy = tf.distribute.MirroredStrategy() 
     
    while not done: 
        print('Iteration:', i) 
         
    ### Feature Extraction ### 
        if i == 0: 
            with strategy.scope():            
                parallel_model_feature = YOPO_feature(image_size)   ### create a new model

        else:                
            with strategy.scope():            
                parallel_model_feature = tf.keras.Model(inputs=parallel_model.input, \
                    outputs=[parallel_model.layers[-2].get_output_at(0), parallel_model.layers[-2].get_output_at(1), parallel_model.layers[-2].get_output_at(2)]) 

        parallel_model_feature.compile(loss='categorical_crossentropy', optimizer='adam')                 
                     
        features = parallel_model_feature.predict([x_train, x_train, x_train])[0]          
        # features=array[15,1,1,1,32]

    ### Feature Clustering ###                              
             
        labels_temp_proba, labels_temp, K, same_K, features_pca, gmm = statistical_fitting(features = features, labels = labels, candidateKs = candidateKs,\
                                                                                            K = K, reg_covar = reg_covar, i = i) 
         
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
               parallel_model_feature.save(model_path)   ### save model here ###
               
               labels_best = labels.copy()   ### save current labels if DDBI improves ###  

                 
               pickle_dump(labels_best, label_path) 
 
 
            DBI_best = DBI                                                                                                                   
                             
        print('DDBI:', DBI, '############################################') 


        

    ### Permute Samples ###             
    
        label_one_hot, x_train_permute, labels_permute = prepare_training_data(x_train = x_train, labels_temp_proba = labels_temp_proba, labels = labels, n = 1)
     
         
    ### Finetune new model with current estimated K ### 
        if not same_K:
            with strategy.scope():                             
                # num_labels = K = 12
                model_classification = YOPO_classification(num_labels=K, vector_size = 32)
                model_classification._name = 'classifier' 
                optimizer = tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)                  
                model_classification.compile(optimizer= optimizer, loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', SNN, NSNN], \
                                             loss_weights = [1, 0, 0, 0, 0])                      
                
                features = np.squeeze(features)

                _features = [features, features, features] # _features 有3个向量，每个都是(15,32)
                _label = [label_one_hot, label_one_hot, label_one_hot, np.zeros(features.shape[0]), np.zeros(features.shape[0])]
                # _label有5个向量，前三个是(15,12)，后两个是(15,)
                model_classification.fit(_features, \
                    _label, \
                        epochs=10, batch_size=batch_size, shuffle=True, verbose = 0) 
                # features.shape=(20,32), label_one_hot=(15,12)
        ### New YOPO ### 
                parallel_model = tf.keras.Model(parallel_model_feature.input, model_classification(parallel_model_feature.output))
             
    ### CNN Training ###           
 
        lr *= 0.95 
        optimizer = tf.keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)          
        parallel_model.compile(optimizer= optimizer, loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy', mse, nmse], loss_weights = [0, 0, 0, 1, 1])
        
        from sklearn.utils import class_weight
        
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels) 



        _label2 = [labels_permute[0], labels_permute[1], labels_permute[2], \
                            np.zeros((x_train_permute[0].shape[0],)), np.zeros((x_train_permute[0].shape[0],))]

        
        # parallel_model.fit(x_train_permute, _label2,\
        #                     epochs=1, batch_size=batch_size, shuffle=True)

        # x_train_permute有三组，shape=(15,24,24,24,1)        
        # labels_permute[0].shape=(15,12)， 前三个为(15,12)，后两个为(15,)
        
        
        
        del x_train_permute
        gc.collect() 
        # 清除内存，尽量避免主动调用gc.collect()
        # 除非当你new出一个大对象，使用完毕后希望立刻回收，释放内存
