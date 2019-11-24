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
# Coherenceモデル
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus


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
    default=32
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
    default=32
)
@click.option(
    '--testing-mode',
    help='テストモードで実行するかどうか (default False).',
    type=bool,
    default=False
)
def main(cuda,batch_size,epochs,top_words,testing_mode):#上のコマンドライン引数
    define_topic = 30 # トピックの数を事前に定義
    hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/hist.txt" , dtype=float)
    label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/label.txt" , dtype=np.int32)
    test_hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_hist.txt" , dtype=float)
    test_label = np.loadtxt( "/home/yoshiwo/workspace/res/study/experiment/make_synthetic_data/test_label.txt" , dtype=np.int32)
    tensor_tr = torch.from_numpy(hist).float()
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    #print("作成したヒストグラム->\n"+str(hist))
    print("hist.shape->{}".format(hist.shape))
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


    ds_train = TensorDataset(torch.from_numpy(hist).float())
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
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=training_callback
    )
    print('Evaluation stage.')
    autoencoder.eval()
    decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
    topics = [
        [reverse_vocab[item.item()] for item in topic]
        for topic in decoder_weight.topk(top_words, dim=0)[1].t()
    ]
    """
    各トピックの単語をテキストファイルに保存
    """
    file_name = "./top_words.txt"
    file = open(file_name, 'w')
    for index, topic in enumerate(topics):
        print(','.join(topic))
        file.write(str(index) + ':' + ','.join(topic) + "\n")
    file.close()

    if not testing_mode:
        writer.add_embedding(
            autoencoder.encoder.linear1.weight.detach().cpu().t(),
            metadata=indexed_vocab,
            tag='feature_embeddings',
        )
    writer.close()

    """
    Perplexityの計算
    学習後のパラメータを用いてデータセット全てに対してPerplexityの計算を行う
    """
    dataloader = DataLoader(
        ds_val,
        batch_size=test_hist.shape[0],
        )
    losses = []
    counts = []
    for index, batch in enumerate(dataloader):
        batch = batch[0]
        print(f"batch->{batch}")
        recon, mean, logvar, z = autoencoder(batch)
        losses.append(autoencoder.loss(batch, recon, mean, logvar).detach().cpu())
        counts.append(batch.sum(1).detach().cpu())
    #print(f"losses->{losses}\ncounts->{counts}")
    losses = losses[0].clone()
    counts = counts[0].clone()
    avg = (losses / counts).mean()
    print('The approximated perplexity is: ', math.exp(avg))


if __name__ == '__main__':
    main()
