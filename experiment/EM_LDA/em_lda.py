import numpy as np
import matplotlib.pyplot as plt
import torch

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

if __name__ == "__main__":
    # 各パラメータの初期化
    K = 3
    W = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr.txt" , dtype=int)
    label = np.loadtxt( "../make_synthetic_data/k"+str(K)+"tr_z.txt" , dtype=np.int32)
    hist = torch.from_numpy(W).float()
    word_sum = int(sum(hist.sum(1)))
    D = W.shape[0]
    V = W.shape[1]
    PER_V = W.shape[1]
    N = np.full(D, PER_V)
    
    # 予測パラメータの初期化
    theta_est = normalized_random_array(D, K)
    phi_est = normalized_random_array(K, V)

    # estimate parameters
    q = np.zeros((D, np.max(N), K))

    print(f"q : {q}")
    print(f"q.shape : {q.shape}")

    T = 200 # エポック数
    plt_epoch_list = np.arange(T) # グラフの横軸
    likelihood = np.zeros(T) # 対数尤度を格納するリスト
    for t in range(T):
        print(t)

        # E step
        for (d, N_d) in enumerate(N):
            q[d, :N_d, :] = normalize(theta_est[d, :] * phi_est[:, W[d]].T, axis = 1)

        # M step
        theta_est[:, :] = normalize(q.sum(axis = 1), axis = 1)
        q_sum = np.zeros((K, V))
        for (d, W_d) in enumerate(W):
            v, index, count = np.unique(W_d, return_index= True, return_counts = True)
            q_sum[:, v[:]] += count[:] * q[d, index[:], :].T
        phi_est[:, :] = normalize(q_sum, axis = 1)

        # likelihood
        for (d, W_d) in enumerate(W):
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
    
    ari = adjusted_rand_score(theta_est.argmax(axis=1),label)
    print(f"ARI->{ari}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-EM(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('emliks.png')
