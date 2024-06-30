import tensorflow as tf
import numpy as np
import numbers
from math import pi


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
      If seed is None, return the RandomState singleton used by np.random.
      If seed is an int, return a new RandomState instance seeded with seed.
      If seed is already a RandomState instance, return it.
      Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                        ' instance' % seed)

class MLP(tf.keras.models.Model):
    """
    2 Layer Multi Layer Precptron Model
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model returns probabilities that are shaped (n, k) if they relate to an individual sample.
    """
    def __init__(self, hidden_dim, optimizer, tol=1e-3 ,mu_init=None, init_params="kmeans", random_state=None):
        """
        The class expects data to be fed as a flat tensor in (n, d).
        Optimizer is used to define the keras optimizer that shall be employed to decreases loss by tuning various parameters and weights, 
        hence minimizing the loss function, providing better accuracy of model faster.
        The class owns:
            x:               tf.Tensor (n, d)
            mu:              tf.Tensor (1, 2, d)
            init_params:     str
            n_feature:       int
            hidden_dim:      int
            optimizer:       keras.optimizer
            tol:             float
            random_state:    int
        args:
            hiddem_dim:      int
            optimizer:       keras.optimizer
        options:
            mu_init:         tf.Tensor (1, k, d)
            var_init:        tf.Tensor (1, k, d) or (1, k, d, d)
            init_params:     str
            tol:             float
            random_state:    int
        """
        super(MLP, self).__init__()
        self.mu_init = mu_init
        self.hidden_dim = hidden_dim
        self.optimizer = optimizer
        self.tol = tol
        self.init_params= init_params
        self.random_state = random_state
        self.loss_value = 0
        self.lst =[]
    
    
    def _init_param(self):
        """
        Intialize the model 
        Model consist of two linear layers, and the output of first layer will be activated by ReLu function before being put in last layer. 
        The size of each layer is nk × d → nk × hidden dimension → nk × 2
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, input_shape=(self.n_features,)),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
        
        if self.mu_init is not None:
            assert self.mu_init.shape == (1, 2,
                                           self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
            2, self.n_features)
            
            self.mu = self.add_weight(name='mu',
                                     shape=self.mu_init.shape,
                                     initializer=tf.initializers.Constant(self.mu_init),
                                     dtype='float32',
                                     trainable=False,
                                     )
        else:
            self.mu = self.add_weight(name='mu', shape=(1, 2, self.n_features), dtype='double',
                                      initializer=tf.initializers.Zeros(),
                                      trainable=False)

    
    def check_size(self,x):
        """
        Input a tensor and expands it dimension along axis=1
        (n, d) --> (n, 1, d)

        arg:
            x:   tf.Tensor (n, d)

        Returns:
            x :  tf.Tensor (n,1,d)
        """
        if len(x.shape)==2:
            x.tf.expand_dims(x,1)
        return x
    
    def loss(self, x, mu, proba):
        """
        The performance of the model is evaluated against and the parameters learned by the modeled output.
        Arg: 
            x:        tf.Tensor(n,d) or (n,1,d)
            mu:       tf.Tensor(1,2,d) 
            proba:    tf.Tensor(n,2)
        Returns:
            loss:     tf.float32

        """
        x =tf.cast(x, tf.float32)
        mu =tf.cast(mu, tf.float32)
        proba =tf.cast(proba, tf.float32)

        if len(x.shape)==3:
            x = tf.squeeze(x, 1)
    
        loss = tf.reduce_mean(tf.math.multiply(tf.reshape(proba[:,0], shape = [x.shape[0],1]),
        tf.pow(tf.subtract(x,tf.tile(mu[:,0,:],(x.shape[0],1))),2))) + \
               tf.reduce_mean(tf.math.multiply(tf.reshape(proba[:,1], shape = [x.shape[0],1]),
        tf.pow(tf.subtract(x,tf.tile(mu[:,1,:],(x.shape[0],1))),2)))
        
        return loss
    
    def _update_mu(self, x, proba):
        """
        Updates mu after every epoch
        arg:
            x:        tf.Tensor(n,d) or (n,1,d)
            proba:    tf.Tensor(n,2)
        return:
            mu:       tf.Tensor(1,2,d)
        """
        if len(x.shape)==3:
            x = tf.squeeze(x, 1)
        
        hard_clus = tf.cast(tf.squeeze(tf.argmax(proba, 1)), tf.int32)
        clus_classes, _ = tf.unique(hard_clus)
        # print(hard_clus)
        # print(clus_classes)
        
        if tf.shape(tf.where(tf.equal(hard_clus,0)))[0] == 0:
            mu0 = tf.convert_to_tensor(np.full((1,self.n_features), 1e-7))
        else:
            sub_class0 = tf.gather(x,indices=[tf.where(tf.equal(hard_clus, 0)).numpy().flatten()])
            mu0 = tf.divide(tf.reduce_sum(sub_class0, axis=1),sub_class0.shape[1])
            
        if tf.shape(tf.where(tf.equal(hard_clus,1)))[0] == 0:
            mu1 = tf.convert_to_tensor(np.full((1,self.n_features), 1e-7))
        else:
            sub_class1 = tf.gather(x,indices=[tf.where(tf.equal(hard_clus, 1)).numpy().flatten()])
            mu1 = tf.divide(tf.reduce_sum(sub_class1, axis=1),sub_class1.shape[1])
            
        return tf.expand_dims(tf.concat([mu0,mu1], 0), axis=0)
        
    
    def fit(self,x, max_epochs=2):
        """
        Fits model to the data.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        options:
            max_epochs: Type ---> int32. It is used to assign the max number of epochs.
        """
        self.n_features = x.shape[-1]
        self._init_param()
        
        input_data = tf.convert_to_tensor(x)
        self.mu = self.get_kmeans_mu(x)
        # self.lst.append(self.mu)
        
        loss_diff = np.inf
        loss_old = 0.0
        
        for epochs in range(max_epochs):
            self.lst.append(self.mu)
            with tf.GradientTape() as tape:
                logits = self.model(input_data, training=True)
                # print((logits))
                # print((input_data))
                self.loss_value = self.loss(input_data,self.mu, logits)
                # print(self.loss_value)
            
            grads = tape.gradient(self.loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
            self.mu = self._update_mu(input_data, logits)
            
            loss_diff = self.loss_value.numpy() - loss_old
            # print(loss_diff)
            # print(np.abs(loss_diff) < self.tol)
            if np.abs(loss_diff) < self.tol:
                # print(loss_diff)
                break
            else:
                loss_old = self.loss_value.numpy()
        
        return self.loss_value.numpy()
                
        
    def _get_mean_and_variance(self, x , proba):
        """
        Extracts the variance of distribution
        arg:
            x:        tf.Tensor(n,d) or (n,1,d)
            proba:    tf.Tensor(n,2)
        return:
            mu:             tf.Tensor (1, 2, d)
            variance:       tf.Tensor (1, 2, d)
        """
        if len(x.shape)==3:
            x = tf.squeeze(x, 1)
        d = x.shape[0]

        hard_clus = tf.cast(tf.squeeze(tf.argmax(proba, 1)), tf.int32)
        clus_classes, _ = tf.unique(hard_clus)
                
        if tf.shape(tf.where(tf.equal(hard_clus,0)))[0] == 0:
            s0 = tf.convert_to_tensor(np.full((1,d,d), 1e-8))
            m0 = tf.convert_to_tensor(np.full((1,self.n_features), 1e-8))
        else:
            sub_class0 = tf.gather(x,indices=[tf.where(tf.equal(hard_clus, 0)).numpy().flatten()])
            d0 = sub_class0.shape[1]
            # print(sub_class0.shape)
            m0 = tf.divide(tf.math.reduce_sum(sub_class0, axis=1),sub_class0.shape[1])
            # print(m0.shape)
            s0 = tf.divide(tf.matmul(tf.transpose(tf.subtract(tf.squeeze(sub_class0,axis=0), m0)),
                                     tf.subtract(tf.squeeze(sub_class0,axis=0), m0)), 
                           tf.cast(d0-1, tf.float64))
            
            s0 = tf.expand_dims(s0, axis=0)


            
        if tf.shape(tf.where(tf.equal(hard_clus,1)))[0] == 0:
            s1 = tf.convert_to_tensor(np.full((1,d,d), 1e-8))
            m1 = tf.convert_to_tensor(np.full((1,self.n_features), 1e-8))
        else:
            sub_class1 = tf.gather(x,indices=[tf.where(tf.equal(hard_clus, 1)).numpy().flatten()])
            d1 = sub_class1.shape[1]
            m1 = tf.divide(tf.math.reduce_sum(sub_class1, axis=1),sub_class1.shape[1])
            s1 = tf.divide(tf.matmul(tf.transpose(tf.subtract(tf.squeeze(sub_class1,axis=0), m1)),
                                     tf.subtract(tf.squeeze(sub_class1,axis=0), m1)), 
                           tf.cast(d1-1, tf.float64))
            
            s1 = tf.expand_dims(s1, axis=0)
            
        return tf.expand_dims(tf.concat([m0,m1], 0), axis=0), tf.expand_dims(tf.concat([s0,s1], 0), axis=0)
    
    def predict(self, x , return_proba = False, return_prior = False, return_mu_and_variance = False):
        """
        Assigns input data to one of the mixture components (2 component mixture model) by evaluating the logits under each.
        If return_proba=True returns normalized probabilities of class membership. 
        If return_prior=True returns pi 
        If return_variance=True returns sample variance of Data
        args:
            x:                 tf.Tensor (n, d) or (n, 1, d)
            return_proba:      bool
            return_prior:      bool
            
        returns:
            p_k:        tf.Tensor (n, 2)
            (or)
            y:          tf.LongTensor (n)
            (or)
            pi:         tf.FloatTensor (1, 2)
            (or)
            variance:   tf.FloatTensor (1, 2, d)
        """
        x = tf.convert_to_tensor(x)
        
        if len(x.shape)==3:
            x = tf.squeeze(x, 1)
        
        logits = self.model(x, training=False)
        
        if return_proba:
            return logits
        
        if return_prior:
            return tf.expand_dims(tf.reduce_mean(logits, axis=0), axis=0)
        
        if return_mu_and_variance:
            return self._get_mean_and_variance(x, logits)
                    
        return tf.cast(tf.squeeze(tf.argmax(logits, 1)), tf.int32)
    
    
    # Not in use (Used only for testing purpose)
    def get_lst(self):
        """
        Returns the list of tensor contaning mu used in every epoch
        """
        return self.lst
    
    def model_param(self, x):
        """
        Returns:
            pi:               tf.Tensor (1, k)
            mu:               tf.Tensor (1, k, d)
            var:              tf.Tensor (1, k, d, d)
        """
        out_mu, out_var = self.predict(x, return_mu_and_variance = True)
        return tf.expand_dims(self.predict(x, return_prior=True),axis=-1) , out_mu, out_var
    
    def pi(self, x):
        """
        Returns pi --> tf.Tensor(1, 2)
        """
        return self.predict(x, return_prior=True)
    
    def mu_and_variance(self, x):
        """
        Returns mu and sample variance ---> tf.Tensor(1, 2, d), tf.Tensor(1, 2, d)
        """
        return self.predict(x, return_mu_and_variance = True)
    
    def hard_clustering(self,x):
        """
        Returns Hard Clustering probabilities of class membership.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            y:          tf.LongTensor (n)
        """
        return self.predict(x)
    
    
    def soft_clustering(self,x):
        """
        Returns Soft Clustering probabilities of class membership.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            y:          tf.LongTensor (n, k+1)
        """
        return self.predict(x, return_proba=True)
    
    def get_kmeans_mu(self, x, n_centers=2, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            tf.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        Returns: 
            mu:           tf.FloatTensor (1, n_centers , d)
        """
        if len(x.shape) == 3:
            x = tf.squeeze(x, 1)
        x_min, x_max = tf.reduce_min(x), tf.reduce_max(x)
        x = (x - x_min) / (x_max - x_min)

        random_state = check_random_state(self.random_state)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x.numpy()[random_state.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - tmp_center, ord=2, axis=2)
            l2_cls = tf.argmin(l2_dis, axis=1)

            cost = 0
            for c in range(n_centers):
                cost += tf.reduce_mean(tf.norm(x[l2_cls == c] - tmp_center[c], ord=2, axis=1))

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf
        while delta > min_delta:
            l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - center, ord=2, axis=2)
            l2_cls = tf.argmin(l2_dis, axis=1)
            center_old = tf.convert_to_tensor(center, dtype=tf.double)

            for c in range(n_centers):
                center[c] = tf.reduce_mean(x[l2_cls == c], axis=0)

            delta = tf.reduce_max(tf.reduce_sum(tf.square(center_old - center), axis=1))

        return tf.expand_dims(center, 0) * (x_max - x_min) + x_min


if  __name__ == '__main__':
    pass