#実行時間の計測
import time
#メイン処理
import glob
import click
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import pickle
# DeepLDA用の訓練用関数とvaeモデル
from ptavitm.model import train
#from ptavitm.vae import ProdLDA
from ptavitm.mavitm import MAVITM
# データローダ
from torch.utils.data import DataLoader
from ptavitm.utils import CountTensorDataset
import math
# Coherenceモデル
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus



def main():#上のコマンドライン引数
    define_topic = 3 # トピックの数を事前に定義
    x1_hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/hist.txt" , dtype=float)
    x2_hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt" , dtype=float)
    x1_label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/label.txt" , dtype=np.int32)
    x2_label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_label.txt" , dtype=np.int32)
    test_hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt" , dtype=float)
    test_label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_label.txt" , dtype=np.int32)
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    #print("作成したヒストグラム->\n"+str(hist))
    print("hist.shape->{}".format(x1_hist.shape))
    print("全単語数{}".format(x1_hist.shape[0]*x1_hist.shape[1]))
    #print("len(hist)->",len(hist[0]))
    x1_vocab = {}
    x2_vocab = {}
    for i in range(len(x1_hist[0])):
        x1_vocab["ID"+str(i)] = i
    for i in range(len(x2_hist[0])):
        x2_vocab["ID"+str(i)] = i
    #print("vocab->",vocab)

    """
    vocab
    {'0番目の特徴': 0,'1番目の特徴':1 }
    BoFの局所特徴量を単語で表現
    BoWと同じように訓練できるようにしただけ
    """
    # ここまでがBoFを作成する作業#############################################
    print('Loading input data')
    x1_reverse_vocab = {x1_vocab[word]: word for word in x1_vocab}
    x1_indexed_vocab = [x1_reverse_vocab[index] for index in range(len(x1_reverse_vocab))]
    x2_reverse_vocab = {x2_vocab[word]: word for word in x2_vocab}
    x2_indexed_vocab = [x2_reverse_vocab[index] for index in range(len(x2_reverse_vocab))]
    ds_train = TensorDataset(torch.from_numpy(x1_hist).float(),torch.from_numpy(x2_hist).float())
    ds_val = TensorDataset(torch.from_numpy(x1_hist).float(),torch.from_numpy(x2_hist).float())

    model = MAVITM(
    #topics=define_topic,
    #input_x1=len(hist[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
    #input_x2=len(hist[0]),
    #hidden1_dimension=100,
    #hidden2_dimension=100,
    topics=define_topic,
    joint_input = len(x1_hist[0])+len(x2_hist[0]),
    input_x1=len(x1_hist[0]),
    input_x2=len(x2_hist[0]),
    hidden1_dimension=100,
    hidden2_dimension=100,
    )
    print(model)
    model.eval()

    train_batch = 10
    trainloader = DataLoader(
        ds_train,
        batch_size=train_batch,
    )
    for x,t in enumerate(trainloader):
        #print(f"X1->{t[0]}")
        #print(f"X2->{t[0]}")
        mean, logvar, x1_mean, x1_logvar, x2_mean, x2_logvar, x1_recon, x2_recon, z_hoge = model(t[0],t[1])




if __name__ == '__main__':
    main()
