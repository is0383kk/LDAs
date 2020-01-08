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
    K = 3
    x1 = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr.txt" , dtype=np.int32)
    x2 = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr.txt" , dtype=np.int32)
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = x1.shape[0]
    V1 = x1.shape[1]
    V2 = x2.shape[1]
    #print("D1,V1->",D1,V1)
    #print("D2,V2->",D2,V2)
    #print("x1",x1)
    N1 = np.full(D, V1)
    N2 = np.full(D, V2)
    #print("N1,N2->",N1,N2)
    alpha0, beta0_x1, beta0_x2 = 1.0, 1.0, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta1 = beta0_x1 + np.random.rand(K, V1)
    beta2 = beta0_x1 + np.random.rand(K, V2)
    theta = normalized_random_array(D, K)
    phi_x1 = normalized_random_array(K, V1)
    phi_x2 = normalized_random_array(K, V2)

    X1 = []
    for (d, N_d) in enumerate(N1):
        x1_data = []
        count = x1[d]
        index = np.where(x1[d]==0)
        count = count[count!=0]
        v = np.arange(len(x1[d]))
        v = np.delete(v, index)
        
        for i1, i2 in enumerate(count):    
            for i3 in range(i2):
                x1_data.append(v[i1])
        X1.append(x1_data)
    
    X2 = []
    for (d, N_d) in enumerate(N1):
        x2_data = []
        count = x2[d]
        index = np.where(x2[d]==0)
        count = count[count!=0]
        v = np.arange(len(x1[d]))
        v = np.delete(v, index)
        
        for i1, i2 in enumerate(count):    
            for i3 in range(i2):
                x2_data.append(v[i1])
        X2.append(x2_data)
    #print("X1->",X1)
        
    #print(f"W->\n{W[0]}")
    #print(f"data->\n{data[0]}")
    #print(f"Z->\n{Z}")
    # estimate parameters
    T = 1
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    
    t1 = time.time() # 処理前の時刻
    for t in range(T):
        print(f"Epoch->{t}")
        
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta1 = digamma(beta1) - digamma(beta1.sum(axis = 1, keepdims = True))
        dig_beta2 = digamma(beta2) - digamma(beta2.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new1 = np.ones((K, V1)) * beta0_x1
        beta_new2 = np.ones((K, V2)) * beta0_x2
        for (d, N_d) in enumerate(N1):
            #W_data = []
            # q
            q1 = np.zeros((V1, K))
            q2 = np.zeros((V2, K))
            q = q1 + q2
            #v, count = np.unique(W[d], return_counts = True)
            #count = W[d]
            
            #index = np.where(W[d]==0)
            
            #count = count[count!=0]
            
            #v = np.arange(len(W[d]))
            #v = np.delete(v, index)
            
            #for i1, i2 in enumerate(count):
            #    
            #    for i3 in range(i2):
            #        W_data.append(v[i1])
            #data.append(W_data)
            #print("data->",data)                
            
            #print(f"count->{count}")
            #print(f"v->{v}")
            v1, count1 = np.unique(X1[d], return_counts = True)
            v2, count2 = np.unique(X2[d], return_counts = True)
            #v1, count2 = np.unique(X2[d], return_counts = True)
            q1[v1, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta1[:, v1]) * count1).T
            q2[v2, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta2[:, v2]) * count2).T
            q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)
            

            # alpha, beta
            alpha_new[d, :] += count1.dot(q1[v1])
            #print("alpha_new->",alpha_new)
            #print("count",count.dot(q[v]))
            #print("q[v]", q[v])
            beta_new1[:, v1] += count1 * q1[v1].T
            
            
            alpha = alpha_new.copy()
            beta1 = beta_new1.copy()
    
            theta_est = np.array([np.random.dirichlet(a) for a in alpha])
            phi_est = np.array([np.random.dirichlet(b) for b in beta1])
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
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
        """
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
    
    
    
