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
    W = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt" , dtype=int)
    hist = torch.from_numpy(W).float()
    word_sum = int(sum(hist.sum(1)))
    print(f"word_sum=>{word_sum}")
    K = 20
    D = W.shape[0]
    PER_V = W.shape[1]
    N = np.full(D, PER_V)
    V = 50
    # 予測パラメータの初期化
    theta_est = normalized_random_array(D, K)
    phi_est = normalized_random_array(K, V)

    #print(f"W : {W}")
    #print(f"W : {W.shape[1]}")

    # estimate parameters
    q = np.zeros((D, np.max(N), K))

    print(f"q : {q}")
    print(f"q.shape : {q.shape}")

    T = 100 # エポック数
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

    # perplexity
    perplexity = np.exp(-likelihood[:] / word_sum ) 
    print(f"Perplexity->{np.exp(-likelihood[-1] / word_sum)}")
    # グラフの保存
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(18,9))

    axL.plot(plt_epoch_list,likelihood)
    axL.set_title('loss',fontsize=23)
    axL.set_xlabel('epoch',fontsize=23)
    axL.set_ylabel('loss',fontsize=23)
    axL.tick_params(labelsize=18)
    axL.grid(True)


    axR.plot(plt_epoch_list,perplexity)
    axR.set_title('perplexity',fontsize=23)
    axR.set_xlabel('epoch',fontsize=23)
    axR.set_ylabel('perplexity',fontsize=23)
    axR.tick_params(labelsize=18)
    axR.grid(True)

    fig.savefig('lp_em.png')
