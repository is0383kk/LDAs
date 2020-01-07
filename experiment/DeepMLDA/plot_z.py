import time
import argparse
#メイン処理
import glob
import numpy as np

import torch
from torch.utils.data import TensorDataset
#from ptavitm.vae import ProdLDA
from ptavitm.vae_tanh import ProdLDA
# データローダ
from torch.utils.data import DataLoader
# クラス推定
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.cluster import adjusted_rand_score

parser = argparse.ArgumentParser(description='Plot latent variable:Amortized MLDA')
parser.add_argument('--k', type=int, default=10, metavar='K',
                    help="トピック数を指定")
args = parser.parse_args()

define_topic = args.k # トピックの数を事前に定義
tr_x1 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x1.txt" , dtype=float)
tr_label = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_z.txt" , dtype=np.int32)
te_x1 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x1.txt" , dtype=float)
te_label = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_z.txt" , dtype=np.int32)

model = ProdLDA(
in_dimension=len(tr_x1[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
hidden1_dimension=100, # 中間層
hidden2_dimension=100,
topics=define_topic
)
model.load_state_dict(torch.load('./deeplda.pth'))
#autoencoder.load_state_dict(torch.load('./deeplda.pth'))

"""
vocab
{'0番目の特徴': 0,'1番目の特徴':1 }
BoFの局所特徴量を単語で表現
BoWと同じように訓練できるようにしただけ
"""
# ここまでがBoFを作成する作業#############################################
print('Loading input data')
#データセット定義
ds_tr = TensorDataset(torch.from_numpy(tr_x1).float(), torch.from_numpy(tr_label).int())
ds_te = TensorDataset(torch.from_numpy(te_x1).float(), torch.from_numpy(te_label).int())
#crossmodal_te = TensorDataset(torch.from_numpy(te_x1).float(), torch.from_numpy(te_label).int())

#モデルの定義

print(f"autoencoder->{model}")

batch = 1000


trainloader = DataLoader(
    ds_tr,
    batch_size=batch,
)

testloader = DataLoader(
    ds_te,
    batch_size=batch,
)

"""
crossmodalloader = DataLoader(
    crossmodal_te,
    batch_size=batch,
)
"""
# 潜在変数の可視化
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
elif define_topic == 20:
    colors = ["#000000", "#808080", "#b0c4de", "#4169e1", "#0000ff", "#00ffff", "#006400", "#8fbc8f", '#00ff7f', '#32cd32',"#556b2f", "#eee8aa", "#ffff00", "#ffa500", "#f4a460", "#8b0000", "#ff00ff", "#ff7f50", '#ff0000', '#f781bf']
elif define_topic == 30:
    colors = ["red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf',"red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf',"red", "green", "blue", "orange", "purple", "yellow", "black", "cyan", '#a65628', '#f781bf']

def visualize_zs(zs, labels, mode, ari):
    #plt.figure(figsize=(10,10))
    fig = plt.figure(figsize=(10,10))
    #ax = Axes3D(fig)
    points = PCA(n_components=2, random_state=0).fit_transform(zs)
    for p, l in zip(points, labels):
        plt.title(f"Latent space:Top:{str(define_topic)}, Doc:{str(batch)}, ARI:{ari}", fontsize=22)
        plt.xlabel("Latent space:xlabel", fontsize=21)
        plt.ylabel("Latent space:ylabel", fontsize=21)
        plt.tick_params(labelsize=17)
        plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
        #ax.scatter(p[0], p[1], p[2], marker="${}$".format(l),c=colors[l])
    plt.savefig(f'./sample_z/{mode}k{str(define_topic)}d{str(te_x1.shape[0])}ari{int(ari*100)}.png')

for x,t in enumerate(trainloader):
    print("***********Joint multi-modal inference***********")
    #print(f"te_x1 ->{t[0]}")
    #print(f"te_x1 ->{t[1]}")
    recon, mean, logvar, z_hoge = model(t[0])
    tr_z = z_hoge.cpu()
    tr_label = t[1].cpu()
    #print(f"te_label->{te_label}")
    predict_tr_label = F.softmax(z_hoge,dim=1).argmax(1).numpy()
    #print(f"tr_label->{predict_tr_label}")
    #print(f"predict_train_label->{predict_train_label}")
    tr_ari = adjusted_rand_score(tr_label,predict_tr_label)
    print(f"Joint:ARI->{tr_ari}")
    #visualize_zs(tr_z.detach().numpy(), tr_label.detach().numpy(), "TRAIN", tr_ari)
    break

score = []
for i in range(30):
    for x,t in enumerate(testloader):
        print("***********Joint multi-modal inference***********")
        #print(f"te_x1 ->{t[0]}")
        #print(f"te_x1 ->{t[1]}")
        recon, mean, logvar, z_hoge = model(t[0])
        te_z = z_hoge.cpu()
        te_label = t[1].cpu()
        #print(f"te_label->{te_label}")
        predict_te_label = F.softmax(z_hoge,dim=1).argmax(1).numpy()
        print(f"pr_label->{predict_te_label}")
        #print(f"predict_train_label->{predict_train_label}")
        te_ari = adjusted_rand_score(te_label,predict_te_label)
        score.append(te_ari)
        print(f"Joint:ARI->{te_ari}")
        #visualize_zs(te_z.detach().numpy(), te_label.detach().numpy(), "TEST", te_ari)
        break
print("平均ARI", sum(score) / len(score))

"""
for x,t in enumerate(crossmodalloader):
    print("***********Cross-modal inference***********")
    t[1] = torch.zeros((test_batch,te_x1.shape[1]))
    #print(t[1])
    mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge = model(t[0], t[1])
    te_z = z_hoge.cpu()
    te_label = t[2].cpu()
    predict_te_label = F.softmax(z_hoge,dim=1).argmax(1).numpy()
    #print(f"predict_train_label->{predict_train_label}")
    te_ari = adjusted_rand_score(te_label,predict_te_label)
    #print(f"TEST:ARI->{te_ari}")
    #visualize_zs_train(train_z.detach().numpy(), predict_train_label)
    #visualize_zs_test(te_z.detach().numpy(), te_label.detach().numpy())
    break
"""
