import numpy as np
from scipy.special import digamma
from sklearn.metrics.cluster import adjusted_rand_score
import time
import pickle
import os

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

def save_model( save_dir, q):
    try:
        os.mkdir( save_dir )
    except:
        pass

    with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
        pickle.dump( [a, b], f )

def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a, b = pickle.load( f )
    return a, b
    
if __name__ == "__main__":

    # initialize parameters
    train_mode = True
    #train_mode = False
    K = 10
    W = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x5.txt" , dtype=np.int32)
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = W.shape[0]
    V = W.shape[1]
    #print("D,V->",D,V)
    #print("W",W)
    N = np.full(D, V)
    alpha0, beta0 = 1.0, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta = beta0 + np.random.rand(K, V)
    #theta = normalized_random_array(D, K)
    #phi = normalized_random_array(K, V)
    
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
    #print("data->",data)

    #print(f"W->\n{W[0]}")
    #print(f"data->\n{data[0]}")
    #print(f"Z->\n{Z}")
    # estimate parameters
    T = 100
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    
    if train_mode == False:
        q_test = load_model( "./learn_result" )
        print(sum(q_test),sum(q_test[0]))
    
    t1 = time.time() # 処理前の時刻
    for t in range(T):
        print(f"Epoch->{t}")

        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        #print("dig_gammma",dig_alpha)
        dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new = np.ones((K, V)) * beta0
        for (d, N_d) in enumerate(N):
            # q
            q = np.zeros((V, K))
            v, count = np.unique(data[d], return_counts = True)
            if train_mode:
                q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
                q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

                # alpha, beta
                alpha_new[d, :] += count.dot(q[v])
                beta_new[:, v] += count * q[v].T
            else:
                q_test = load_model( "./learn_result" )
                q_test[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
                q_test[v, :] /= q_test[v, :].sum(axis = 1, keepdims = True)
                alpha_new[d, :] += count.dot(q_test[v])
                beta_new[:, v] += count * q_test[v].T
        
        alpha = alpha_new.copy()
        beta = beta_new.copy()

        theta_est = np.array([np.random.dirichlet(a) for a in alpha])
        phi_est = np.array([np.random.dirichlet(b) for b in beta])
        
        t3 = time.time()
        elapsed_time = t3-t1
        print(f"経過時間：{elapsed_time}")
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        print(f"ARI->{ari}")
        """
        # 対数尤度計算
        for (d, W_d) in enumerate(data):
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
        """
    if train_mode:
        save_model( "./learn_result", dig_alpha, dig_beta )
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
