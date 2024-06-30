import torch

# import  tensorflow as tf
# 检查CUDA是否可用
print(torch.cuda.is_available())
 

import tensorflow as tf
 
print(tf.test.is_gpu_available())