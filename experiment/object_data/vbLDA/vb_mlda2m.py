import numpy as np
from scipy.special import digamma
from sklearn.metrics.cluster import adjusted_rand_score
import time

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

def make_data(N, hist):
    X = []
    for (d, N_d) in enumerate(N):
        x_data = []
        count = hist[d]
        index = np.where(hist[d]==0)
        count = count[count!=0]
        v = np.arange(len(hist[d]))
        v = np.delete(v, index)

        for i1, i2 in enumerate(count):
            for i3 in range(i2):
                x_data.append(v[i1])
        X.append(x_data)
    return X


if __name__ == "__main__":

    # initialize parameters
    K = 5
    data = []
    #x1 = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x1.txt" , dtype=np.int32) 
    #x2 = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x2.txt" , dtype=np.int32)
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x1.txt" , dtype=np.int32) )
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x2.txt" , dtype=np.int32) )
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = data[0].shape[0]
    alpha0, betax1, betax2 = 0.1, 0.1, 0.1
    alpha = alpha0 + np.random.rand(D, K)
    theta = normalized_random_array(D, K)
    V = []
    N = []
    X = []
    beta = []
    for i in range(2):
        V.append(data[i].shape[1])
        N.append(np.full(D,V[i]))
        X.append(make_data(N[i], data[i]))
    beta1 = betax1 + np.random.rand(K, V[0])    
    beta2 = betax2 + np.random.rand(K, V[1])
   
        
    
    """
    for (d, N_d) in enumerate(N1):
        x_data = []
        count = x1[d]
        index = np.where(x1[d]==0)
        count = count[count!=0]
        v = np.arange(len(x1[d]))
        v = np.delete(v, index)

        for i1, i2 in enumerate(count):
            for i3 in range(i2):
                x_data.append(v[i1])
        X1.append(x_data)
    """ 
    

    #print(f"W->\n{W[0]}")
    #print(f"data->\n{data[0]}")
    #print(f"Z->\n{Z}")
    # estimate parameters
    T = 100
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)

    t1 = time.time() # 処理前の時刻
    for t in range(T):
        print(f"Epoch->{t}")

        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta1 = digamma(beta1) - digamma(beta1.sum(axis = 1, keepdims = True))
        dig_beta2 = digamma(beta2) - digamma(beta2.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new1 = np.ones((K, V[0])) * betax1
        beta_new2 = np.ones((K, V[1])) * betax2
        #for m in range(len(data[0]))
        for (d, N_d) in enumerate(N[0]):
            print("Nd->",N_d)
            #W_data = []
            # q
            q = np.zeros((V[0], K)) + np.zeros((V[1], K))
            v1, count1 = np.unique(X[0][d], return_counts = True)
            v2, count2 = np.unique(X[1][d], return_counts = True)
            q[v1, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta1[:, v1]) * count1).T
            q[v2, :] += (np.exp(dig_beta2[:, v2]) * count2).T
            q[v1, :] /= q[v1, :].sum(axis = 1, keepdims = True)
            q[v2, :] /= q[v2, :].sum(axis = 1, keepdims = True)


            # alpha, beta
            alpha_new[d, :] += count1.dot(q[v1]) + count2.dot(q[v2])
            beta_new1[:, v1] += count1 * q[v1].T
            beta_new2[:, v2] += count2 * q[v2].T

        alpha = alpha_new.copy()
        beta1 = beta_new1.copy()
        beta2 = beta_new2.copy()

        theta_est = np.array([np.random.dirichlet(a) for a in alpha])
        phi_est1 = np.array([np.random.dirichlet(b) for b in beta1])
        phi_est2 = np.array([np.random.dirichlet(b) for b in beta2])
        t3 = time.time()
        elapsed_time = t3-t1
        print(f"経過時間：{elapsed_time}")
        #print(f"phi_est->\n{phi_est}")
        #print(f"theta_est->\n{theta_est.argmax(axis=1)}")
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        print(f"ARI->{ari}")
          

        """
        # 対数尤度計算
        for (d, W_d) in enumerate(X1):
            #print(f"log_W_d->{W_d}")
            #print(f"log_phi_est->\n{phi_est[:, W_d]}")
            #print(f"log_phi_est0->\n{phi_est[:, 2]}")
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est1[:, W_d])).sum()
        
        
        for (d, W_d) in enumerate(X2):
            #print(f"log_W_d->{W_d}")
            #print(f"log_phi_est->\n{phi_est[:, W_d]}")
            #print(f"log_phi_est0->\n{phi_est[:, 2]}")
            likelihood[t] += np.log(phi_est2[:, W_d]).sum()
       """
    print(sum(q[0]))      
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
