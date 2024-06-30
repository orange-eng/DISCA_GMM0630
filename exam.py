
# the process always shut down suddenly, and there is no bug information.
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


for i in range(len(infos)):
    _info = infos[i][0]
    print('_info=',_info)