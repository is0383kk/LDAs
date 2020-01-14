import numpy as np
from scipy.special import digamma

def normalize(ndarray, axis):
    return ndarray / ndarray.sum(axis = axis, keepdims = True)

def normalized_random_array(d0, d1):
    ndarray = np.random.rand(d0, d1)
    return normalize(ndarray, axis = 1)

if __name__ == "__main__":

    # initialize parameters
    D, K, V = 3, 3, 3
    alpha0, beta0 = 0.1, 1.0
    alpha = alpha0 + np.random.rand(D, K)
    beta = beta0 + np.random.rand(K, V)
    theta = normalized_random_array(D, K)
    phi = normalized_random_array(K, V)
    #print("alpha \n",alpha)

    # for generate documents
    _theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
    _phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T
    
    #for v in range(V):
    #   print("phi[:, :v+1].sum(axis = 1)\n",phi[:, :v+1].sum(axis = 1))
    #print("theta\n", _theta)
    #print("phi\n", _phi)
    # generate documents
    W, Z = [], []
    N = np.random.randint(1, 5, D)
    #N = np.full(D, V)
    for (d, N_d) in enumerate(N):
        Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
        W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis = 1))
    print("W->",W)
    # estimate parameters
    T = 1
    plt_epoch_list = np.arange(T)
    likelihood = np.zeros(T)
    for t in range(T):
        print(f"{t}回目")
        dig_alpha = digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))
        dig_beta = digamma(beta) - digamma(beta.sum(axis = 1, keepdims = True))

        alpha_new = np.ones((D, K)) * alpha0
        beta_new = np.ones((K, V)) * beta0
        for (d, N_d) in enumerate(N):
            # q
            q = np.zeros((V, K))
            v, count = np.unique(W[d], return_counts = True)
            print(f"W[{d}]->{W[d]}")
            print(f"v->{v}, count->{count}")
            q[v, :] = (np.exp(dig_alpha[d, :].reshape(-1, 1) + dig_beta[:, v]) * count).T
            q[v, :] /= q[v, :].sum(axis = 1, keepdims = True)

            # alpha, beta
            alpha_new[d, :] += count.dot(q[v])
            beta_new[:, v] += count * q[v].T

        alpha = alpha_new.copy()
        beta = beta_new.copy()

        theta_est = np.array([np.random.dirichlet(a) for a in alpha])
        phi_est = np.array([np.random.dirichlet(b) for b in beta])
        
        # 対数尤度計算
        for (d, W_d) in enumerate(W):
            print(f"log_W_d->{W_d}")
            print(f"log_phi_est->\n{phi_est[:, W_d]}")
            print(f"log_phi_est0->\n{phi_est[:, 2]}")
            likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()
            
    print("theta_est\n", theta_est)
    print("phi_est\n", phi_est)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('LDA-vb(Topic='+ str(K) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,likelihood)

    plt.savefig('liks.png')
