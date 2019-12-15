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
from ptavitm.vae_tanh import ProdLDA
# データローダ
from torch.utils.data import DataLoader
from ptavitm.utils import CountTensorDataset
import math

"""
コマンドライン引数
"""
@click.command()
@click.option(
    '--cuda',
    help='CUDAを使用するかどうか (default False).',
    type=bool,
    default=False
)
@click.option(
    '--batch-size',
    help='バッチサイズ(文書数/batch_size ).',
    type=int,
    default=16
)
@click.option(
    '--epochs',
    help='学習エポック (default 5).',
    type=int,
    default=100
)
@click.option(
    '--top-words',
    help='各トピックにおいて表示するトップ単語の数 (default 12).',
    type=int,
    default=3
)
@click.option(
    '--testing-mode',
    help='テストモードで実行するかどうか (default False).',
    type=bool,
    default=False
)
def main(cuda,batch_size,epochs,top_words,testing_mode):#上のコマンドライン引数
    define_topic = 3 # トピックの数を事前に定義
    hist = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}trc.txt" , dtype=float)
    label = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}trc_label.txt" , dtype=np.int32)
    test_hist = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}tec.txt" , dtype=float)
    test_label = np.loadtxt( f"../make_synthetic_data/k{str(define_topic)}tec_label.txt" , dtype=np.int32)
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    #print("作成したヒストグラム->\n"+str(hist))
    print("hist.shape->{}".format(hist.shape))
    print(f"hist[0]->{hist[0]}")
    print("全単語数{}".format(hist.shape[0]*hist.shape[1]))
    #print("len(hist)->",len(hist[0]))
    vocab = {}
    for i in range(len(hist[0])):
        vocab["ID"+str(i)] = i
    print("vocab->",vocab)
    """
    vocab
    {'0番目の特徴': 0,'1番目の特徴':1 }
    BoFの局所特徴量を単語で表現
    BoWと同じように訓練できるようにしただけ
    """
# ここまでがBoFを作成する作業#############################################
    print('Loading input data')
    reverse_vocab = {vocab[word]: word for word in vocab};
    indexed_vocab = [reverse_vocab[index] for index in range(len(reverse_vocab))]
# ここから本番######################################################################
    writer = SummaryWriter()  # create the TensorBoard object
    """
    トレーニング中に呼び出すコールバック関数，スコープからライターを使用
    """
    def training_callback(autoencoder, epoch, lr, loss, perplexity):
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'perplexity': perplexity,
        }, global_step=epoch)
        decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
        topics = [
            [reverse_vocab[item.item()] for item in topic]
            for topic in decoder_weight.topk(top_words, dim=0)[1].t()
        ]



    #################################################################################


    ds_train = TensorDataset(torch.from_numpy(hist).float(),torch.from_numpy(label).int())
    ds_val = TensorDataset(torch.from_numpy(test_hist).float())


    autoencoder = ProdLDA(
        in_dimension=len(hist[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
        hidden1_dimension=100, # 中間層
        hidden2_dimension=100,
        topics=define_topic
    )
    if cuda:
        autoencoder.cuda()
    print(autoencoder)
    #import pdb; pdb.set_trace()
    ############################################################################
    """
    訓練
    """
    print('Training stage.')
    ae_optimizer = Adam(autoencoder.parameters(), 0.001, betas=(0.99, 0.999))

    train(
        ds_train,
        define_topic,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=training_callback
    )

    print('Evaluation stage.')
    """
    各トピックの単語をテキストファイルに保存
    """
    writer.close()

    """
    Perplexityの計算
    学習後のパラメータを用いてデータセット全てに対してPerplexityの計算を行う
    """
    """
    autoencoder.eval()
    dataloader = DataLoader(
        ds_train,
        batch_size=1,
        )
    for index, batch in enumerate(dataloader):
        #print(f"batch->{batch[0][0][0]}")
        batch = batch[0]
        #print(f"batch->{batch}")
        recon, mean, logvar, z_hoge = autoencoder(batch)
        loss = autoencoder.loss(batch, recon, mean, logvar,define_topic)
        print(f"loss->{loss}")
    """

if __name__ == '__main__':
    main()
