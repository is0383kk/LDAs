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

def save_model( save_dir, model):
    try:
        os.mkdir( save_dir )
    except:
        pass

    with open( os.path.join( save_dir,"model.pickle" ), "wb" ) as f:
        pickle.dump( q, f )

def save_model( save_dir, model):
    try:
        os.mkdir( save_dir )
    except:
        pass

    with open( os.path.join( save_dir,"model.pickle" ), "wb" ) as f:
        pickle.dump( q, f )

def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        q = pickle.load( f )

    return q

if __name__ == "__main__":

    # initialize parameters
    K = 10
    train_mode = True
    #train_mode = False
    print(train_mode)
    save_dir = "./learn_result"
    data = []
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x1.txt" , dtype=np.int32) )
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x2.txt" , dtype=np.int32) )
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x3.txt" , dtype=np.int32) )
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x4.txt" , dtype=np.int32) )
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = data[0].shape[0]
    alpha0, betax1, betax2, betax3, betax4 = 0.1, 0.5, 0.5, 0.5, 0.5
    alpha = alpha0 + np.random.rand(D, K)
    
    V = []
    N = []
    X = []
    beta = []
    for i in range(len(data)):
        V.append(data[i].shape[1])
        N.append(np.full(D,V[i]))
        X.append(make_data(N[i], data[i]))
    beta1 = betax1 + np.random.rand(K, V[0])    
    beta2 = betax2 + np.random.rand(K, V[1])
    beta3 = betax3 + np.random.rand(K, V[2])
    beta4 = betax4 + np.random.rand(K, V[3])
    

    
    
    
    # 変分推論
    T = 10000
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    t1 = time.time() # 処理前の時刻
    if train_mode == False:
        q_test, d_a, d_b1, d_b2 = load_model( "./learn_result" )
        print(d_a) 
    
    for t in range(T):
        print(f"Epoch->{t}")
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta1 = digamma(beta1) - digamma(beta1.sum(axis = 1, keepdims = True))
        dig_beta2 = digamma(beta2) - digamma(beta2.sum(axis = 1, keepdims = True))
        dig_beta3 = digamma(beta3) - digamma(beta3.sum(axis = 1, keepdims = True))
        dig_beta4 = digamma(beta4) - digamma(beta4.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new1 = np.ones((K, V[0])) * betax1
        beta_new2 = np.ones((K, V[1])) * betax2
        beta_new3 = np.ones((K, V[2])) * betax3
        beta_new4 = np.ones((K, V[3])) * betax4
        
        for (d1, N2_d) in enumerate(N[0]):
            q = np.zeros((V[0]+V[1]+V[2], K)) 
            v1, count1 = np.unique(X[0][d1], return_counts = True)
            v2, count2 = np.unique(X[1][d1], return_counts = True)
            v3, count3 = np.unique(X[2][d1], return_counts = True)
            v4, count4 = np.unique(X[3][d1], return_counts = True)
            q[v1, :] = (np.exp(dig_alpha[d1, :].reshape(-1, 1) + dig_beta1[:, v1]) * count1).T
            q[v1, :] /= q[v1, :].sum(axis = 1, keepdims = True)
            q[v2, :] += (np.exp(dig_alpha[d1, :].reshape(-1, 1) + dig_beta2[:, v2]) * count2).T
            q[v2, :] /= q[v2, :].sum(axis = 1, keepdims = True)
            q[v3, :] += (np.exp(dig_alpha[d1, :].reshape(-1, 1) + dig_beta3[:, v3]) * count3).T
            q[v3, :] /= q[v3, :].sum(axis = 1, keepdims = True)
            q[v4, :] += (np.exp(dig_alpha[d1, :].reshape(-1, 1) + dig_beta3[:, v4]) * count4).T
            q[v4, :] /= q[v4, :].sum(axis = 1, keepdims = True)
            alpha_new[d1, :] += count1.dot(q[v1])
            alpha_new[d1, :] += count2.dot(q[v2])
            alpha_new[d1, :] += count3.dot(q[v3])
            alpha_new[d1, :] += count4.dot(q[v4])
            beta_new1[:, v1] += count1 * q[v1].T
            beta_new2[:, v2] += count2 * q[v2].T
            beta_new3[:, v3] += count3 * q[v3].T
            beta_new4[:, v4] += count4 * q[v4].T
       
            
    
            
    
              
        alpha = alpha_new.copy()
        beta1 = beta_new1.copy()
        beta2 = beta_new2.copy()
        beta3 = beta_new3.copy()
        beta4 = beta_new4.copy()
        
        
        

           
        theta_est = np.array([np.random.dirichlet(a) for a in alpha])
        phi_est1 = np.array([np.random.dirichlet(b) for b in beta1])
        phi_est2 = np.array([np.random.dirichlet(b) for b in beta2])
        phi_est3 = np.array([np.random.dirichlet(b) for b in beta3])
        phi_est4 = np.array([np.random.dirichlet(b) for b in beta4])
        t3 = time.time()
        elapsed_time = t3-t1
        
        print("theta",theta_est.argmax(axis=1))
        print(f"経過時間：{elapsed_time}")
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        print(f"ARI->{ari}")
        

    if train_mode:
        save_model( "./learn_result", q)
        print(sum(q),sum(q[0]))
    else:
        print(sum(q_test),sum(q_test[0]))
          
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
