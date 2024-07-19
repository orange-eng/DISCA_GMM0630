# DISCA_GMM

This repository uses pyotrch to reproduce DISCA_GMM_CAVI.

It's based on the tf version of DISCA_GMM, so I do not remove the original tf files.

torch_DISCA_gmmu.py is equal to DISCA_gmmu_cavi_llh_scanning_new.py, and it can run smoothly.

Now, I am working for torch_DISCA_gmmuv2.py, which is equal to disca_gmmu_cavi_llh_hist_new.py.

You just focus on the folder "torchv2".


# 0719

In folder "torchv2", I successfully run v7_crossentropy.py to train a good classifier.

There are some important things:

1) Please normalize the data
2) Crossentropy loss is the best choice
3) I simplify the structure of YOPO



#### Clone project
```
git clone https://github.com/orange-eng/DISCA_GMM0630.git
```
#### Environments

- python 3.9.12
- torch 1.13.0

All dependencies can be found in requirements.txt. There are some important dependencies.
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mrcfile h5py scikit-learn scikit-image matplotlib scipy==1.5.4 pypng
```
#### training
```
cd /code/DISCA_GMM/torchv2
python v7_crossentropy.py --M 1 --sub_epoch 10 --subtomo_num 7000 --subtomo_num_test 7000 --hidden_num 32
```

If you want to use MSELoss as loss function, you can run:
```
cd /code/DISCA_GMM/torchv2
python v7_mse.py --M 1 --sub_epoch 10 --subtomo_num 7000 --subtomo_num_test 7000 --hidden_num 32
```


#### Visualization
```
python visualization_NN_GMMv2.py --M 1 --sub_epoch 10 --subtomo_num 7000 --subtomo_num_test 7000 --hidden_num 32 --saving_path '/data/zfr888/EMD_4603/Results7/'
```