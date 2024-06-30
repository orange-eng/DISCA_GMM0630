import numpy as np


# #数据
# # 随机种子
# np.random.seed(0)

# # 假设男生的身高均值是1.71 标准差为 0.056
# man_mean = 1.71
# man_sigma = 0.056

# # 数据个数
# num = 10000
# # 男生数据
# rand_data_man = np.random.normal(man_mean, man_sigma, num)
# # 标签 设男生的标签为1
# y_man = np.ones(num)

# # 女生的身高均值1.58 标准差 0.051
# np.random.seed(0)
# women_mean = 1.58
# women_sigma = 0.051
# rand_data_women = np.random.normal(women_mean, women_sigma, num)
# y_women = np.zeros(num)

# # 将数据混合
# data = np.append(rand_data_man, rand_data_women)
# data = data.reshape(-1, 1)
# # print(data)
# y = np.append(y_man, y_women)
# # print(y)


# #模型
# from sklearn.mixture import GaussianMixture
# #n_components 有多少种类 max_iter迭代多少次
# model = GaussianMixture(n_components=2,max_iter=1000)

# model.fit(data)
# print('pai:',model.weights_[0])
# print('mean:',model.means_)
# print('方差:',model.covariances_)

# #预测
# from sklearn.metrics import accuracy_score
# y_pred = model.predict(data)

# print('y_pred=',y_pred)
# # print(accuracy_score(y,y_pred))




# x_train_origin.shape=(15,24,24,24,1)
# label_temp_proba.shape=(15,12)
# labels.shape=(15,)

# n代表着重复生成data的次数
#    label_one_hot = labels_temp_proba


# n=1
# labels = [3,1,5,4]
# labels_tile = np.tile(labels, n) # m*n
# # labels_tile.shape=(15,)

# labels_temp_proba = np.array([[0,0,0,1,0,0],
#                     [0,1,0,0,0,0],
#                     [0,0,0,0,0,1],
#                     [0,0,0,0,1,0]
# ])
# # labels_temp_proba = np.array(labels_temp_proba)

# labels_proba_tile = np.tile(labels_temp_proba, (n, 1)) # (m*n)*n m*n的组合在列重复n次
# # labels_proba_tile.shape=(15,12)
# labels_np = []
# # np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表
# for i in range(len(np.unique(labels))):
#     _flag = labels_proba_tile[:, i]
#     _flag3 = labels_tile != i
#     _flag2 = _flag[_flag3]
#     npi = np.maximum(0, 0.5 - _flag2) # 逐元素比较， 非 i th cluster的样本第i个cluster的概率
#     print('npi=',npi)


# index_negative = [1,0,2,3,4]
# # _neg = np.zeros((5,24,24,24,1))# _neg.shape=(15,24,24,24,1)
# # _neg = np.random.randint(5,3)

# _neg = np.random.randint(low=0, high=10, size=(5, 3))

# x_train_augmented_neg = _neg[index_negative]

# print('x_train_augmented_neg=',x_train_augmented_neg.shape)



import torch

# resylt = torch.tensor([5.5,3], dtype=torch.float64)

# print('resylt=',resylt)




# N = 43

# new_lambda_phi_n0 = torch.ones(size=[14])

# _res1 = torch.reshape(new_lambda_phi_n0, (1,14 ))

# results = torch.tile(torch.reshape(new_lambda_phi_n0, (1,14 )), (N,1))


# print('results=',results.shape)


# s1 = np.random.dirichlet((10, 5, 3), 20).transpose()



# _random = np.array([1.0] * (13+1))  # shape=(14,), k=13
# _random2 = np.random.dirichlet(_random, 64)


# s1 = np.random.dirichlet(_random, 64)
# print('s1.shape=',s1.shape)


# import tensorflow as tf
# #矩阵乘法
# a = torch.tensor([[1,2],[3,4]])
# b = torch.tensor([[2,0],[0,2]])

# import torch
# # create a tensor with shape (3, 4)
# x = torch.randn(3, 4)
# # compute the softmax of the tensor along the last dimension
# y = torch.softmax(x, dim=-1)
# # print the original and softmaxed tensors
# print('x=',x.shape)
# print('y=',y.shape)


# [912,912,456]

factor = 40
tom = np.ones(shape=(928,928,464))

tom = tom[0:912,0:912,0:456]

# adding_pre = 12       # adding_pre=12
# adding_post = 11     # adding_post=11
# # tom.shape=[928,928,464], factor=10
# x_interval_start, y_interval_start, z_interval_start = \
#     [np.array(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor))) for i in range(3)]                        

# x_interval_end, y_interval_end, z_interval_end = \
#     [np.array(list(range(adding_pre, tom.shape[i]-adding_post, int(tom.shape[i]/factor)))[1:] + [tom.shape[i]]) for i in range(3)]   

# print('x_interval_start=',x_interval_start)
# print('x_interval_end=',x_interval_end)


# subvolumes = []   
# for i in range(factor): 
#     for j in range(factor):           
#         for k in range(factor):       
#             subvolume = tom[x_interval_start[i]: x_interval_end[i], y_interval_start[j]: y_interval_end[j], z_interval_start[k]: z_interval_end[k]]   
#             _vol = np.expand_dims(np.array(subvolume), [0,-1]) 
#             subvolumes.append(_vol)



#从912开始最合适，总共是928

x_interval_start = np.linspace(0, 72, num=4).astype(int)
x_interval_end = np.linspace(24, 96, num=4).astype(int)
print('x_interval_start=',x_interval_start)
print('x_interval_end=',x_interval_end)


z_interval_start = np.linspace(0, 432, num=19).astype(int)
z_interval_end = np.linspace(24, 456, num=19).astype(int)
print('z_interval_start=',z_interval_start)
print('z_interval_end=',z_interval_end)
