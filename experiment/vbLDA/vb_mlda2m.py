import numpy as np
from scipy.special import digamma
from sklearn.metrics.cluster import adjusted_rand_score
import time
import pickle
import os
import matplotlib.pyplot as plt

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


def save_model( save_dir, a, b1, b2, m, k):
    try:
        os.mkdir( save_dir )
    except:
        pass

    with open( os.path.join( save_dir,"modelM"+str(m)+"K"+str(k)+".pickle" ), "wb" ) as f:
        pickle.dump( [a, b1, b2], f )

def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a, b1, b2 = pickle.load( f )

    return a, b1, b2

if __name__ == "__main__":

    # initialize parameters
    K = 30
    train_mode = True
    #train_mode = False
    print(train_mode)
    save_dir = "./learn_result"
    data = []
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x1.txt" , dtype=np.int32) )
    data.append(np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_x2.txt" , dtype=np.int32) )
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    D = data[0].shape[0]
    alpha0, betax1, betax2 = 0.5, 2.5, 2.5
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
    

    
    
    
    # 変分推論
    T = 5000
    plt_epoch_list = np.arange(T)
    ari_list = []
    likelihood = np.zeros(T)
    t1 = time.time() # 処理前の時刻
    
    if train_mode==False:
        dig_alpha, dig_beta1, dig_beta2 = load_model( "./learn_result" )
    
    for t in range(T):
        print(f"m{len(data)}k{K}Epoch->{t}")
        if train_mode:
            dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
            dig_beta1 = digamma(beta1) - digamma(beta1.sum(axis = 1, keepdims = True))
            dig_beta2 = digamma(beta2) - digamma(beta2.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new1 = np.ones((K, V[0])) * betax1
        beta_new2 = np.ones((K, V[1])) * betax2
        
        for (d1, N1_d) in enumerate(N[0]):    
            q1 = np.zeros((V[0], K)) 
            v1, count1 = np.unique(X[0][d1], return_counts = True)
            q1[v1, :] = (np.exp(dig_alpha[d1, :].reshape(-1, 1) + dig_beta1[:, v1]) * count1).T
            q1[v1, :] /= q1[v1, :].sum(axis = 1, keepdims = True)
            alpha_new[d1, :] += count1.dot(q1[v1])
            beta_new1[:, v1] += count1 * q1[v1].T
            
        for (d2, N2_d) in enumerate(N[1]):    
            q2 = np.zeros((V[1], K)) 
            v2, count2 = np.unique(X[1][d2], return_counts = True) 
            q2[v2, :] = (np.exp(dig_alpha[d2, :].reshape(-1, 1) + dig_beta2[:, v2]) * count2).T
            q2[v2, :] /= q2[v2, :].sum(axis = 1, keepdims = True)
            alpha_new[d2, :] += count2.dot(q2[v2])
            beta_new2[:, v2] += count2 * q2[v2].T       
            
            
    
              
        alpha = alpha_new.copy()
        beta1 = beta_new1.copy()
        beta2 = beta_new2.copy()
        
           
        theta_est = np.array([np.random.dirichlet(a) for a in alpha])
        phi_est1 = np.array([np.random.dirichlet(b) for b in beta1])
        phi_est2 = np.array([np.random.dirichlet(b) for b in beta2])
        
        if train_mode:
            save_model( "./learn_result", dig_alpha, dig_beta1, dig_beta2, len(data), K)
        
        t3 = time.time()
        elapsed_time = t3-t1
       
        print("theta",theta_est.argmax(axis=1))
        ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
        ari_list.append(ari)
        print(f"経過時間：{elapsed_time}")
        print(f"ARI->{ari}")
        
        
        # 対数尤度計算
        for (d, W_d) in enumerate(X[0]):
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est1[:, W_d])).sum()
        for (d, W_d) in enumerate(X[1]):
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est2[:, W_d])).sum()
        
        if (t+1)%250 == 0:
            #print(plt_epoch_list[:t+1])
            #print(ari_list)
            plt.figure(figsize=(15,9))
            plt.tick_params(labelsize=18)
            plt.title('MLDA-vb(M=' + str(len(data))+'K='+str(K)+'T='+str(int(elapsed_time))+'):ARI='+str(ari),fontsize=22)
            plt.xlabel('Epoch',fontsize=22)
            plt.ylabel('ARI',fontsize=22)
            plt.plot(plt_epoch_list[:t+1],ari_list)
            plt.savefig('./ari/vb_m'+str(len(data))+'k'+str(K)+"e"+str(t)+'ari.pdf')
        
        if (t+1)%300 == 0:
            plt.figure(figsize=(15,9))
            plt.tick_params(labelsize=18)
            plt.title('MLDA-vb(M=' + str(len(data))+'K='+str(K)+'):Log likelihood',fontsize=22)
            plt.xlabel('Epoch',fontsize=22)
            plt.ylabel('Log likelihood',fontsize=22)
            plt.plot(plt_epoch_list[:t+1],likelihood[:t+1])
            plt.savefig('./liks/vb_m'+str(len(data))+'k'+str(K)+"e"+str(t)+'liks.pdf')
