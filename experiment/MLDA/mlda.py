import numpy as np
import random
import math
import pickle
import os
import matplotlib.pyplot as plt
import time
import argparse
from sklearn.metrics.cluster import adjusted_rand_score

parser = argparse.ArgumentParser(description='MLDA')
parser.add_argument('--k', type=int, default=30, metavar='K',
                    help="トピック数を指定")
args = parser.parse_args()

# ハイパーパラメータ
__alpha = 0.5
__beta = 3.0
epoch_num = 100 # 学習エポック

def calc_lda_param( docs_mdn, topics_mdn, K, dims ):
    M = len(docs_mdn)
    D = len(docs_mdn[0])

    # 各物体dにおいてトピックzが発生した回数
    n_dz = np.zeros((D,K))

    # 各トピックzにおいて特徴wが発生した回数
    n_mzw = [ np.zeros((K,dims[m])) for m in range(M)]

    # 各トピックが発生した回数
    n_mz = [ np.zeros(K) for m in range(M) ]

    # 数え上げ処理
    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            N = len(docs_mdn[m][d])    # 物体に含まれる特徴数
            for n in range(N):
                w = docs_mdn[m][d][n]       # 物体dのn番目の特徴インデックス
                z = topics_mdn[m][d][n]     # 特徴に割り当てられているトピック
                n_dz[d][z] += 1
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1

    return n_dz, n_mzw, n_mz


def sample_topic( d, w, n_dz, n_zw, n_z, K, V ):
    P = [ 0.0 ] * K

    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zw[:,w] + __beta) / (n_z[:] + V *__beta)
    for z in range(1,K):
        P[z] = P[z] + P[z-1]

    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z



# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数分for文を回す
            doc.append(v)
    return doc

# 尤度計算関数
def calc_liklihood( data, n_dz, n_zw, n_z, K, V  ):
    lik = 0

    P_wz = (n_zw.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( np.sum(n_dz[d]) + K *__alpha )
        Pwz = Pz * P_wz
        Pw = np.sum( Pwz , 1 ) + 0.000001
        lik += np.sum( data[d] * np.log(Pw) )

    return lik

def save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, m, k ):
    try:
        os.mkdir( save_dir )
    except:
        pass

    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt=str("%f") )

    for m in range(M):
        Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dims[m] *__beta)
        Pdw = Pdz.dot(Pwz.T)
        np.savetxt( os.path.join( save_dir, "Pmdw[%d].txt" % m ) , Pdw )

    with open( os.path.join( save_dir, "modelM"+str(m)+"K"+str(k)+".pickle" ), "wb" ) as f:
        pickle.dump( [n_mzw, n_mz], f )


def load_model( load_dir , m, k):
    model_path = os.path.join( load_dir, "modelM"+str(m)+"K"+str(k)+".pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# MLDAのメイン処理
def mlda( data, label , K, num_itr=epoch_num, save_dir="model", load_dir=None ):
    plt_epoch_list = np.arange(int(num_itr))
    # 尤度のリスト
    liks = []

    M = len(data)       # モダリティの数
    print(f"トピック数:{K}")
    print(f"モダリティの数:{M}")
    dims = []
    for m in range(M):
        if data[m] is not None:
            dims.append( len(data[m][0]) )
            D = len(data[m])    # 物体の数
        else:
            dims.append( 0 )

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_mdn = [[ None for i in range(D) ] for m in range(M)]
    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m] is not None:
                docs_mdn[m][d] = conv_to_word_list( data[m][d] )
                topics_mdn[m][d] = np.random.randint( 0, K, len(docs_mdn[m][d]) ) # 各単語にランダムでトピックを割り当てる

    # LDAのパラメータを計算
    n_dz, n_mzw, n_mz = calc_lda_param( docs_mdn, topics_mdn, K, dims )

    # 認識モードでは学習したパラメータを読み込む
    if load_dir:
        n_mzw, n_mz = load_model( load_dir )
    t1 = time.time() # 処理前の時刻
    ari_list = []
    for it in range(num_itr):
        print(str(it + 1) + "回目")
        # メイン処理
        for d in range(D):
            for m in range(M):
                if data[m] is None:
                    continue

                N = len(docs_mdn[m][d]) # 物体dのモダリティmに含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]       # 特徴のインデックス
                    z = topics_mdn[m][d][n]     # 特徴に割り当てられているカテゴリ


                    # データを取り除きパラメータの更新
                    n_dz[d][z] -= 1

                    if not load_dir:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1

                    # サンプリング
                    z = sample_topic( d, w, n_dz, n_mzw[m], n_mz[m], K, dims[m] )

                    # データをサンプリングされたクラスに追加してパラメータを更新
                    topics_mdn[m][d][n] = z
                    n_dz[d][z] += 1

                    if not load_dir:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1

        lik = 0
        for m in range(M):
            if data[m] is not None:
                lik += calc_liklihood( data[m], n_dz, n_mzw[m], n_mz[m], K, dims[m] )
        liks.append( lik )
       
        t3 = time.time()       
        #plot( n_dz, liks, D, K )
        elapsed_time = t3-t1
        # 経過時間を表示
        
        doc_dopics = np.argmax( n_dz , 1 )
        #doc_dopics = doc_dopics[::-1]
        #print(f"doc_dopics -> {doc_dopics}")
        #print(f"label -> {label[0]}")
        ari = adjusted_rand_score(doc_dopics,label[0])
        ari_list.append(ari)
        print("ari->",ari)    
        file_name = "./time.txt"
        try:
            file = open(file_name, 'a')
            file.write(str(elapsed_time)+"\n")
        except Exception as e:
            print(e)
        finally:
            file.close()
        print(f"k{K}m{len(data)}経過時間：{elapsed_time}")
    
        save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, M,K )
    
        np_liks = np.array(liks)
        if (it+1)%50 == 0:
            #print(plt_epoch_list[:t+1])
            #print(ari_list)
            plt.figure(figsize=(15,9))
            plt.tick_params(labelsize=18)
            plt.title('MLDA-cgb(M=' + str(len(data))+'K='+str(K)+'T='+str(int(elapsed_time))+'):ARI='+str(ari),fontsize=22)
            plt.xlabel('Epoch',fontsize=22)
            plt.ylabel('ARI',fontsize=22)
            plt.plot(plt_epoch_list[:it+1],ari_list)
            plt.savefig('./ari/gb_m'+str(len(data))+'k'+str(K)+"e"+str(it)+'ari.pdf')
        
        if (it+1)%100 == 0:
            plt.figure(figsize=(17,9))
            plt.tick_params(labelsize=18)
            plt.title('MLDA-cgb(M=' + str(len(data))+'K='+str(K)+'):Log likelihood',fontsize=22)
            plt.xlabel('Epoch',fontsize=24)
            plt.ylabel('Log likelihood',fontsize=24)
            plt.plot(plt_epoch_list[:it+1],np_liks)
            #plt.show()
            plt.savefig('./liks/cgb_m'+str(len(data))+'k'+str(K)+"e"+str(it)+'liks.pdf')
        
    
    
    
def main():
    topic = args.k
    data = []
    label = []
    data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_x1.txt" , dtype=np.int32) )
    data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_x2.txt" , dtype=np.int32) )
    #data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_x3.txt" , dtype=np.int32) )
    #data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_x4.txt" , dtype=np.int32) )
    #data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_x5.txt" , dtype=np.int32) )
    #data.append( np.loadtxt( "../make_synthetic_data/k"+str(topic)+"te_x6.txt" , dtype=np.int32) )
    label.append(np.loadtxt( "../make_synthetic_data/k"+str(topic)+"tr_z.txt" , dtype=np.int32))
    #for i in range(30):
    mlda( data, label , topic, 5000, "learn_result" )

    #data[1] = None
    #mlda( data, label , topic, 20, "recog_result" , "learn_result" )


if __name__ == '__main__':
    main()
