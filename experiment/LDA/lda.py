import time
import numpy as np
from scipy.sparse import load_npz
import random
import math
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics.cluster import adjusted_rand_score


#ハイパーパラメータ
__alpha = 1.0
__beta = 0.3



def calc_lda_param( docs_dn, topics_dn, K, V ):
    D = len(docs_dn)

    n_dz = np.zeros((D,K))   # 各文書Dにおいてトピックzが発生した回数
    n_zv = np.zeros((K,V))   # 各トピックzにおいて単語vが発生した回数
    n_z = np.zeros(K)        # 各トピックが発生した回数

    # 数え上げる
    for d in range(D):
        N = len(docs_dn[d])    # 文書に含まれる単語数
        for n in range(N):
            v = docs_dn[d][n]  # ドキュメントDのn番目の単語インデックス
            z = topics_dn[d][n]     # 単語に割り当てられるトピック
            n_dz[d][z] += 1
            n_zv[z][v] += 1
            n_z[z] += 1

    #print("n_dz->"+str(n_dz))
    """
    n_dz = [[3. 2. 1.]]　
    文書1においてトピック1：3,トピック2:2,トピック3:3
    """

    #print("n_zv->"+str(n_zv))
    """
    n_zv = [[1. 1. 0. 0. 2. 1. 0. 0. 1. 0. 1. 1. 1. 1.]]
    トピック１においてi番目の単語が出現した回数
    """

    print("n_z->"+str(n_z))
    """
    z = [10 6 7]
    トピック１は10回
    トピック2は6回
    トピック3は7回
    """
    return n_dz, n_zv, n_z


def sample_topic( d, v, n_dz, n_zv, n_z, K, V ):
    P = [ 0.0 ] * K

    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zv[:,v] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    """
    語彙の総数=14
    bow =
    [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1.] x 10
    """
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
    #語彙の総数分 v：14(0123...13)
        for n in range(data[v]): # 語彙の発生した回数分for文を回す
            # 語彙 x 発生回数
            # [2,3,11,12,13] x 10
            # [2 2 2 2 2 2 2 2 2 2, 3 ... 3, 13...]
            doc.append(v)
    return doc

# 尤度計算
def calc_liklihood( data, n_dz, n_zv, n_z, K, V  ):
    lik = 0

    # 上の処理を高速化
    P_vz = (n_zv.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( np.sum(n_dz[d]) + K *__alpha )
        Pvz = Pz * P_vz
        Pv = np.sum( Pvz , 1 ) + 0.000001
        lik += np.sum( data[d] * np.log(Pv) )

    return lik

def save_model(save_dir, n_dz, n_zv, n_z ):
    try:
        os.mkdir( save_dir )
    except:
        pass

    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T

    Pzv = n_zv + __beta
    Pzv = (Pzv.T / Pzv.sum(1)).T

    np.savetxt( "Pdz.txt", Pdz, fmt=str("%f") )
    np.savetxt( "Pzv.txt", Pzv, fmt=str("%f") )

    with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
        pickle.dump( [n_zv, n_z], f )

def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# ldaのメイン関数
def lda( data , label, K , epoch =100, save_dir="model", load_dir=None):
    epoch_num = epoch
    t1 = time.time() # 処理前の時刻
    plt.ion()
    # 尤度のリスト
    liks = []

    # 単語の種類数
    V = len(data[0])    # 語彙数(語彙の総数)
    D = len(data)       # 文書数
    #print(V)
    #print(D)
    num_p = sum(data[0])*D # perplexityの分母
    print("num_p->",num_p)
    print("語彙数: " + str(V) + ",文書数: " + str(D))

    # data内の単語を一列に並べる　（計算しやすくするため）
    docs_dn = [ None for i in range(D) ]
    topics_dn = [ None for i in range(D) ]

    for d in range(D):
        docs_dn[d] = conv_to_word_list( data[d] )
        topics_dn[d] = np.random.randint( 0, K, len(docs_dn[d]) ) # 各単語にランダムでトピックを割り当てる

    """
    d = np.array(docs_dn[3])
    print("ddn",d.shape) 50
    t = np.array(topics_dn[3])
    print("tdn",t.shape) 50

    BoW =
    [[1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
    [1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 1. 0. 0. 0.]
    [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1.]] x 10倍してる
    ###############################################
    出現した語彙（のインデックス）を一列にしたものddn[3]: [2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 11 11 11 11 11 11 11 11 11 11 12 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13]出現した語彙のインデックス x 10個（最初の読み込みの部分）
    ###############################################
    tdn[3]: [1 1 1 2 1 1 1 0 1 0 2 0 2 1 1 2 1 1 2 1 1 0 2 0 2 1 0 0 1 1 0 2 0 0 0 1 2 1 1 1 0 0 2 0 1 2 2 1 1 1]ddnに対応するトピックをランダム割当て
    """



    # LDAのパラメータを計算
    n_dz, n_zv, n_z = calc_lda_param( docs_dn, topics_dn, K, V )

    if load_dir is not None:
        n_zv, n_z = load_model( load_dir )
        
    for it in range(epoch_num):
        print(str(it + 1) + "回目")
        # メインの処理
        for d in range(D): # d：文書のインデックス（４つなら0~3）
            N = len(docs_dn[d]) # 文書Dに含まれる単語数
            for n in range(N):
                v = docs_dn[d][n]       # 単語のインデックス
                z = topics_dn[d][n]     # 単語に割り当てられているトピック


                # データを取り除きパラメータを更新
                # n 番目の単語 v (トピック z)についてカウンタを減算
                n_dz[d][z] -= 1
                if load_dir is None:
                    n_zv[z][v] -= 1
                    n_z[z] -= 1

                # サンプリング
                z = sample_topic( d, v, n_dz, n_zv, n_z, K, V )

                # データをサンプリングされたクラスに追加してパラメータを更新
                topics_dn[d][n] = z
                n_dz[d][z] += 1
                if load_dir is None:
                    n_zv[z][v] += 1
                    n_z[z] += 1


        lik = calc_liklihood( data, n_dz, n_zv, n_z, K, V )
        liks.append( lik )
        #print ("対数尤度" + str(lik))
        doc_dopics = np.argmax( n_dz , 1 )
        #print ("分類結果" + str(doc_dopics))
        
        #print("---------------------")
        #print("n_dz->",n_dz)
        """
        # グラフ表示
        plt.clf()
        #plt.subplot("121")
        #plt.title( "P(z|d)" )
        #plt.imshow( n_dz / np.tile(np.sum(n_dz,1).reshape(D,1),(1,K)) , interpolation="none" )
        plt.subplot("121")
        plt.title( "liklihood" )
        plt.plot( range(len(liks)) , liks )
        plt.draw()
        plt.pause(0.1)
        """
    t2 = time.time()
    # 経過時間を表示
    elapsed_time = t2-t1
    file_name = "./10k_time.txt"
    np_liks = np.array(liks)
    #print(np_liks)
    try:
        file = open(file_name, 'a')
        file.write(str(elapsed_time)+"\n")
    except Exception as e:
        print(e)
    finally:
        file.close()
    print(f"経過時間：{elapsed_time}")
    
    # テスト時はperplexity
    ari = adjusted_rand_score(doc_dopics,label)
    print("doc_topic",doc_dopics)
    print("test_label",label)
    print("ARI=>",ari)
    if load_dir:
        file_name = "./5k_ari_simple.txt"
        perplexity = np.exp(-np_liks[:] / num_p)
        try:
            file = open(file_name, 'a')
            file.write(str(ari)+"\n")
        except Exception as e:
            print(e)
        finally:
            file.close()
    else:
        perplexity = np.full(len(liks), -1)
    
    
    
    plt_epoch_list = np.arange(epoch_num)
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(18,9))

    np.save('./loss_list.npy', np_liks)
    np.save('./perplexity.npy', perplexity)
    plt_loss_list = np.load('./loss_list.npy')
    plt_perplexity = np.load('./perplexity.npy')

    axL.plot(plt_epoch_list,plt_loss_list)
    axL.set_title('loss',fontsize=23)
    axL.set_xlabel('epoch',fontsize=23)
    axL.set_ylabel('loss',fontsize=23)
    axL.tick_params(labelsize=18)
    axL.grid(True)

    axR.plot(plt_epoch_list,plt_perplexity)
    axR.set_title('perplexity',fontsize=23)
    axR.set_xlabel('epoch',fontsize=23)
    axR.set_ylabel('perplexity',fontsize=23)
    axR.tick_params(labelsize=18)
    axR.grid(True)
    
    fig.savefig('lda_lp.png')
    
    #plt.savefig('result.png')
    if load_dir is None:
        save_model(save_dir, n_dz, n_zv, n_z )
    #plt.ioff()
    #plt.show()
    #print(docs_dn[3])
    #print(topics_dn[3])

def main():
    topic = 10 # トピック数を指定
    test = True
    if test == False:
        root = "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/hist.txt"
        label_data = "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/label.txt"
        #n = 10 # データの水増し用の変数
        #data = np.loadtxt( root , dtype=np.int32)*n # 発生回数にnをかけて水増し可能
        data = np.loadtxt( root , dtype=np.int32)
        label = np.loadtxt( label_data , dtype=np.int32)
        #print(data)
        #for i in range(30):
        lda( data , label, topic, 50, "learn_result" )
    else:
        root = "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt"
        label_data = "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_label.txt"
        #n = 10 # データの水増し用の変数
        #data = np.loadtxt( root , dtype=np.int32)*n # 発生回数にnをかけて水増し可能
        data = np.loadtxt( root , dtype=np.int32)
        label = np.loadtxt( label_data , dtype=np.int32)
        #print(data)
        #for i in range(30):
        lda( data, label, topic,  50, "recog_result" , "learn_result" )

if __name__ == '__main__':
    main()
