#実行時間の計測

#メイン処理
import glob
import numpy as np

import torch
from torch.utils.data import TensorDataset
from ptavitm.vae import ProdLDA
# データローダ
from torch.utils.data import DataLoader

define_topic = 3 # トピックの数を事前に定義
test_hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt" , dtype=float)
test_label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_label.txt" , dtype=np.int32)

"""
vocab
{'0番目の特徴': 0,'1番目の特徴':1 }
BoFの局所特徴量を単語で表現
BoWと同じように訓練できるようにしただけ
"""
# ここまでがBoFを作成する作業#############################################
print('Loading input data')
vocab = {}
for i in range(len(test_hist[0])):
    vocab["ID"+str(i)] = i
reverse_vocab = {vocab[word]: word for word in vocab};
indexed_vocab = [reverse_vocab[index] for index in range(len(reverse_vocab))]
#データセット定義
ds_val = TensorDataset(torch.from_numpy(test_hist).float(),torch.from_numpy(test_label).int())
#モデルの定義
autoencoder = ProdLDA(
    in_dimension=len(test_hist[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
    hidden1_dimension=50, # 中間層
    hidden2_dimension=50,
    topics=define_topic
)

print("autoencoder->{}".format(autoencoder))

autoencoder.load_state_dict(torch.load('./deeplda.pth'))
"""
各文書の潜在変数を可視化してクラスタリング
"""
test_batch = 1000
dataloader = DataLoader(
    ds_val,
    batch_size=test_batch,
)


#print("decoder_weight->\n"+str(decoder_weight.t()))
#print("decoder_weight.shape->\n"+str(decoder_weight.t().shape))
#print("decoder_weight.topk->\n"+str(decoder_weight.topk(top_words, dim=0)[1].t()))

autoencoder.eval()
import time
for i in range(30):
    for x,t in enumerate(dataloader):
        t1 = time.time()
        """
        x:インデックス（使わない）
        t[0]:文書
        t[1]:人口データを元に付けた文書ラベル
        """
        #print("t[0]->",t[0])
        #print("label->",t[1])
        #print("autoencoder(t[0])->",autoencoder.encode(t[0]))
        #print("autoencoder(t[0])->",autoencoder(t[0]))
        #a, b, c = autoencoder.encode(Variable(t[0], volatile=True))
        encoded, mean, logvar = autoencoder.encode(t[0]) # 訓練後の潜在変数の抽出
        #print(mean,logvar)
        t2 = time.time()
        elapsed_time = t2 - t1
        # 書き込むファイルのパスを宣言する
        file_name = "./dldain_time.txt"

        try:
            file = open(file_name, 'a')
            file.write(str(elapsed_time)+"\n")
        except Exception as e:
            print(e)
        finally:
            file.close()
        print(f"経過時間：{elapsed_time}")
