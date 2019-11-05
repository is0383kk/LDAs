import numpy as np
import matplotlib.pyplot as plt

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

if __name__ == "__main__":

    # 各パラメータの初期化
    D = 10 # 文書数
    K = 3 # トピック数
    V = 5 # 単語数
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)
    theta_est = normalized_random_array(D, K)
    phi_est = normalized_random_array(K, V)

    # 文書生成のためのパラメータ
    _theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T


    # 文書の生成
    W, Z = [], []
    N = np.random.randint(1, 10, D) # 各文書の単語数
    """
    N: [2,3,4]
    W: [array([])]
    """
    print(f"N : {N}")
    print(f"N.shape : {N.shape}")
    print(f"np.max(N) : {np.max(N)}")
    for (d, N_d) in enumerate(N):
        Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis = 1))



    #W = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/hist.txt" , dtype=float)
    print(f"W : {W}")

    # estimate parameters
    q = np.zeros((D, np.max(N), K))
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
    perplexity = np.exp(-likelihood[:] / N.sum())

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
