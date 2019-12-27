#実行時間の計測

#メイン処理
import glob
import numpy as np

import torch
from torch.utils.data import TensorDataset
from ptavitm.vae_tanh import ProdLDA
# データローダ
from torch.utils.data import DataLoader
import time


define_topic = 3 # トピックの数を事前に定義
z_dim = 64
hist = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}trc.txt" , dtype=float)
label = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}trc_label.txt" , dtype=np.int32)
test_hist = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}tec.txt" , dtype=float)
test_label = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}tec_label.txt" , dtype=np.int32)

autoencoder = ProdLDA(
    in_dimension=len(hist[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
    hidden1_dimension=100, # 中間層
    hidden2_dimension=100,
    z_dim = z_dim
)
autoencoder.load_state_dict(torch.load('deeplda.pth'))

"""
vocab
{'0番目の特徴': 0,'1番目の特徴':1 }
BoFの局所特徴量を単語で表現
BoWと同じように訓練できるようにしただけ
"""
# ここまでがBoFを作成する作業#############################################
print('Loading input data')
#データセット定義
ds_train = TensorDataset(torch.from_numpy(hist).float(),torch.from_numpy(label).int())
ds_val = TensorDataset(torch.from_numpy(test_hist).float(),torch.from_numpy(test_label).int())
#モデルの定義

print("autoencoder->{}".format(autoencoder))

test_batch = 1000
trainloader = DataLoader(
    ds_train,
    batch_size=test_batch,
)

testloader = DataLoader(
    ds_train,
    batch_size=test_batch,
)
# 潜在変数の可視化
from sklearn.manifold import TSNE
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score as ar

#colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
if define_topic == 2:
    colors = ["red", "green", "blue"]
elif define_topic == 3:
    colors = ["red", "green", "blue"]
elif define_topic == 4:
    colors = ["red", "green", "blue", "orange"]
elif define_topic == 5:
    colors = ["red", "green", "blue", "orange", "purple"]
elif define_topic == 10:
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']

def visualize_zs_train(zs, labels):
    #plt.figure(figsize=(10,10))
    fig = plt.figure(figsize=(10,10))
    #ax = Axes3D(fig)
    points = TSNE(n_components=2, random_state=0).fit_transform(zs)
    for p, l in zip(points, labels):
        plt.title("Latent space (Topic:"+str(define_topic)+", Doc:"+str(test_batch)+", Words:"+str(hist.shape[1])+")", fontsize=24)
        plt.xlabel("Latent space:xlabel", fontsize=21)
        plt.ylabel("Latent space:ylabel", fontsize=21)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=60)
        #ax.scatter(p[0], p[1], p[2], marker="${}$".format(l),c=colors[l])
    plt.savefig('./sample_z/'+'TRAIN'+'k'+str(define_topic)+'v'+str(hist.shape[1])+'d'+str(hist.shape[0])+'.png')

def visualize_zs_test(zs, labels):
    #plt.figure(figsize=(10,10))
    fig = plt.figure(figsize=(10,10))
    #ax = Axes3D(fig)
    points = TSNE(n_components=2, random_state=0).fit_transform(zs)
    for p, l in zip(points, labels):
        plt.title("Latent space (Topic:"+str(define_topic)+", Doc:"+str(test_batch)+", Words:"+str(test_hist.shape[1])+")", fontsize=24)
        plt.xlabel("Latent space:xlabel", fontsize=21)
        plt.ylabel("Latent space:ylabel", fontsize=21)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=60)
        #ax.scatter(p[0], p[1], p[2], marker="${}$".format(l),c=colors[l])
    plt.savefig('./sample_z/'+'TEST'+'k'+str(define_topic)+'v'+str(test_hist.shape[1])+'d'+str(test_batch)+'.png')

for x,t in enumerate(trainloader):
    """
    x:インデックス（使わない）
    t[0]:文書
    t[1]:人口データを元に付けた文書ラベル
    """
    recon, mean, logvar, z = autoencoder(t[0]) # 訓練後の潜在変数の抽出

    z = z.cpu()
    z_label = t[1].cpu()
    visualize_zs_train(z.detach().numpy(), z_label.cpu().detach().numpy())
    break

for x,t in enumerate(testloader):
    """
    x:インデックス（使わない）
    t[0]:文書
    t[1]:人口データを元に付けた文書ラベル
    """
    t1 = time.time()
    recon, mean, logvar, z = autoencoder(t[0]) # 訓練後の潜在変数の抽出
    t2 = time.time()
    # 経過時間を表示
    elapsed_time = t2-t1
    print("実行時間",elapsed_time)
    file_name = "./5k_time.txt"

    try:
        file = open(file_name, 'a')
        file.write(str(elapsed_time)+"\n")
    except Exception as e:
        print(e)
    finally:
        file.close()

    z = z.cpu()
    z_label = t[1].cpu()
    """
    潜在変数zの確認
    """
    visualize_zs_test(z.detach().numpy(), z_label.cpu().detach().numpy())
    break
