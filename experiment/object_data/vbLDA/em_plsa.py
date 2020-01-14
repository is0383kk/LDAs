import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import time

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

if __name__ == "__main__":

    # initialize parameters
    K = 3
    W = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr.txt" , dtype=np.int32)
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = W.shape[0]
    V = W.shape[1]
    N = np.full(D, V)
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)
    theta_est = normalized_random_array(D, K)
    phi_est = normalized_random_array(K, V)

    # for generate documents
    _theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T

    # generate documents
    _W, _Z = [], []
    #N = np.random.randint(100, 300, D)
    
    for (d, N_d) in enumerate(N):
        _Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        _W.append((np.random.rand(N_d, 1) < _phi[_Z[-1], :]).argmax(axis = 1))
    
    data = []
    for (d, N_d) in enumerate(N):
        W_data = []
        # q
        q = np.zeros((V, K))
        #v, count = np.unique(W[d], return_counts = True)
        count = W[d]
        
        index = np.where(W[d]==0)
        
        count = count[count!=0]
        
        v = np.arange(len(W[d]))
        v = np.delete(v, index)
        for i1, i2 in enumerate(count):    
            for i3 in range(i2):
                W_data.append(v[i1])
        data.append(W_data)
        print("data->",data)   
    
    #print("W",_W)
            
    # estimate parameters
    q = np.zeros((D, np.max(N), K))
    T = 300
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    t1 = time.time()
    for t in range(T):
        print(f"epoch->{t}")

        # E step
        for (d, N_d) in enumerate(N):
            q[d, :N_d, :] = normalize(theta_est[d, :] * phi_est[:, data[d]].T, axis = 1)

        # M step
        theta_est[:, :] = normalize(q.sum(axis = 1), axis = 1)
        q_sum = np.zeros((K, V))
        for (d, W_d) in enumerate(data):
            v, index, count = np.unique(W_d, return_index= True, return_counts = True)
            #print("v",v)
            #print("index",index)
            #print("count",count)
            q_sum[:, v[:]] += count[:] * q[d, index[:], :].T
        phi_est[:, :] = normalize(q_sum, axis = 1)
        
        # likelihood
        for (d, W_d) in enumerate(data):
           likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
        
        ################################
        t3 = time.time()
        elapsed_time = t3-t1
        print(f"経過時間：{elapsed_time}")
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        print(f"ARI->{ari}")
        
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-EM(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
