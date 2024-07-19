# 2024/7/12
# without scanning
# Chengzhi Cao

# v4将yaml文件里面的超参数都分开写进代码了，方便调试.
# v4使用了crossEntropyLoss,删除了MSELoss
# v4接下来需要加上positive和negative的数据进行对比训练,使用nn.CosineSimilarity
# v5加入了scanning部分，对每次iteration的数据进行修改


import os, h5py, math
from sklearn.cluster import MeanShift, estimate_bandwidth
from tqdm import *
import sys
sys.path.append('/code/DISCA_GMM')
from disca_dataset.DISCA_visualization import *
# from hist_filtering.filtering import *
from config import *
from torch_DISCA_gmmu import *

from GMMU.torch_gmmu_cavi_stable_new import TORCH_CAVI_GMMU as GMM

import numpy as np
import scipy
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
import skimage,os

import faulthandler
faulthandler.enable()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
#dataset options
parser.add_argument("--filtered_data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/data.h5')
parser.add_argument("--data_path", type=str, default='/data/zfr888/EMD_4603/data_emd4603/original')

#stored path
parser.add_argument("--saving_path", type=str, default='/data/zfr888/EMD_4603/Results5/')
parser.add_argument("--algorithm_name", type=str, default='gmmu_cavi_llh_hist')
parser.add_argument("--filtered_particle_saving_path", type=str, default='/data/zfr888/EMD_4603/Results5/filtered_particle')

parser.add_argument("--image_size", type=int, default=24)
parser.add_argument("--input_size", type=int, default=24)
parser.add_argument("--candidateKs", default=[7,8,9,10,11])

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--factor", default=4)
parser.add_argument("--lr", default=2)
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
parser.add_argument("--DIVIDE", type=int,default=1)
parser.add_argument("--M", type=int,default=1)
parser.add_argument("--sub_epoch", type=int,default=1)
parser.add_argument("--subtomo_num",type=int, default=100)
parser.add_argument("--subtomo_num_test",type=int, default=10)
args = parser.parse_args()



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
        
batch_size = args.batch_size
scanning_num = args.scanning_num ### the number of scanning ###
factor = args.factor ### the num of scanning division###
M = args.M   ### number of iterations ###
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
 
############################################################################################
############################################################################################
############################################################################################
############################################################################################

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




### scanning ###  
# building the subtomo #   

submodel1 = YOLO_Model()
submodel2 = YOPO_classification(num_labels=10, vector_size = 32)
parallel_model = CombinedModel(submodel1,submodel2)
parallel_model = parallel_model.to(device)
        
parallel_model_feature = torch.load(model_path)
model_classification = torch.load(classification_model_path)

submodel1.load_state_dict(parallel_model_feature)
submodel2.load_state_dict(model_classification)

scanning_model = CombinedModel(submodel1,submodel2)


with torch.no_grad():

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
            subt = subt.to(device)
            subt = subt.permute(0,4,1,2,3)

            subt_label = scanning_model(subt) # subt=[1,1,46,46,34],sub_label=[1,4]
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.unsqueeze(subt_label, dim=-1)
            subt_label = torch.squeeze(subt_label,dim=0)
            subt_label = subt_label.cpu().detach().numpy()
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

        print('x_con_start=',x_con_start)
        print('x_con_end=',x_con_end)
        print('x_interval_start=',x_interval_start)
        

        for i in tqdm(range(len(x_interval_start))):
            for j in range(len(y_interval_start)):
                for k in range(len(z_interval_start)):
                    pp_subvolume = pp_map_filtered_labels[  x_con_start[i]: x_con_end[i], \
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
        print('particle_filtered=',len(particle_filtered))
        pp_index = np.concatenate(particle_filtered)
        #save_png(cub_img(pp_map_non_noise[:, :, ::20])['im'], '/local/scratch/v_yijian_bai/disca/deepgmmu/disca/v.png')

        pp_indexs.append(pp_index)
    
    print('pp_indexs=',len(pp_indexs[0]))