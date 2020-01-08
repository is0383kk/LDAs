import numpy as np
from scipy.special import digamma
from sklearn.metrics.cluster import adjusted_rand_score
import time

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

if __name__ == "__main__":

    # initialize parameters
    K = 10
    W = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr.txt" , dtype=np.int32)
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = W.shape[0]
    V = W.shape[1]
    print("D,V->",D,V)
    print("W",W)
    N = np.full(D, V)
    alpha0, beta0 = 1.0, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta = beta0 + np.random.rand(K, V)
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)

    # for generate documents
    _theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T
    
    _Z = []
    _W = [] 
    #N = np.full(D, V)
    for (d, N_d) in enumerate(N):
        _Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        _W.append((np.random.rand(N_d, 1) < _phi[_Z[-1], :]).argmax(axis = 1))

    """
    # generate documents
    #W, Z = [], []
    #N = np.random.randint(100, 300, D)
    
    for (d, N_d) in enumerate(N):
        Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis = 1))
    """
    
        
    #print(f"W->\n{W[0]}")
    #print(f"data->\n{data[0]}")
    #print(f"Z->\n{Z}")
    # estimate parameters
    T = 100
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    data = []
    t1 = time.time() # 処理前の時刻
    for t in range(T):
        print(f"Epoch->{t}")
        
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new = np.ones((K, V)) * beta0
        for (d, N_d) in enumerate(N):
            #W_data = []
            # q
            q = np.zeros((V, K))
            #v, count = np.unique(W[d], return_counts = True)
            count = W[d]
            
            index = np.where(W[d]==0)
            
            count = count[count!=0]
            
            v = np.arange(len(W[d]))
            v = np.delete(v, index)
            
            #for i1, i2 in enumerate(count):
            #    
            #    for i3 in range(i2):
            #        W_data.append(v[i1])
            #data.append(W_data)
            #print("data->",data)                
            
            #print(f"count->{count}")
            #print(f"v->{v}")
            q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
            q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

            # alpha, beta
            alpha_new[d, :] += count.dot(q[v])
            beta_new[:, v] += count * q[v].T
            
            
            alpha = alpha_new.copy()
            beta = beta_new.copy()
    
            theta_est = np.array([np.random.dirichlet(a) for a in alpha])
            phi_est = np.array([np.random.dirichlet(b) for b in beta])
        t3 = time.time()
        elapsed_time = t3-t1
        print(f"経過時間：{elapsed_time}")
        #print(f"phi_est->\n{phi_est}")
        #print(f"theta_est->\n{theta_est.argmax(axis=1)}")
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        print(f"ARI->{ari}")
        """
        # 対数尤度計算   
        for (d, W_d) in enumerate(_W):
            #print(f"log_W_d->{W_d}")
            #print(f"log_phi_est->\n{phi_est[:, W_d]}")
            #print(f"log_phi_est0->\n{phi_est[:, 2]}")
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
        """
    t2 = time.time()
    # 経過時間を表示
    elapsed_time = t2-t1
    file_name = "./time.txt"
    try:
        file = open(file_name, 'a')
        file.write(str(elapsed_time)+"\n")
    except Exception as e:
        print(e)
    finally:
        file.close()
    print(f"経過時間：{elapsed_time}")
    
    
    
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
    
    
    
