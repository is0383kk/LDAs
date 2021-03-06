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
from ptavitm.model_6m import train
#from ptavitm.vae import ProdLDA
from ptavitm.mavitm_6m import MAVITM
# データローダ
from torch.utils.data import DataLoader
from ptavitm.utils import CountTensorDataset
import math

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
    default=200
)
@click.option(
    '--top-words',
    help='各トピックにおいて表示するトップ単語の数 (default 12).',
    type=int,
    default=5
)
@click.option(
    '--testing-mode',
    help='テストモードで実行するかどうか (default False).',
    type=bool,
    default=False
)
@click.option(
    '--k',
    help='トピック数を指定',
    type=int,
    default=10
)

def main(cuda,batch_size,epochs,top_words,testing_mode,k):#上のコマンドライン引数
    define_topic = k # トピックの数を事前に定義
    tr_x1 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x1.txt" , dtype=float)
    tr_x2 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x2.txt" , dtype=float)
    tr_x3 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x3.txt" , dtype=float)
    tr_x4 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x4.txt" , dtype=float)
    tr_x5 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x5.txt" , dtype=float)
    tr_x6 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_x6.txt" , dtype=float)
    #tr_label = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"tr_z.txt" , dtype=np.int32)
    te_x1 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x1.txt" , dtype=float)
    te_x2 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x2.txt" , dtype=float)
    te_x3 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x3.txt" , dtype=float)
    te_x4 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x4.txt" , dtype=float)
    te_x5 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x5.txt" , dtype=float)
    te_x6 = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_x6.txt" , dtype=float)
    #te_label = np.loadtxt( "../make_synthetic_data/k"+str(define_topic)+"te_z.txt" , dtype=np.int32)
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    #print("作成したヒストグラム->\n"+str(hist))
    print("tr_x1.shape->{}".format(tr_x1.shape))
    print("tr_x2.shape->{}".format(tr_x2.shape))
    print("tr_x3.shape->{}".format(tr_x3.shape))
    print("tr_x4.shape->{}".format(tr_x4.shape))
    print("te_x1.shape->{}".format(te_x1.shape))
    print("te_x2.shape->{}".format(te_x2.shape))
    print("te_x3.shape->{}".format(te_x3.shape))
    print("te_x4.shape->{}".format(te_x4.shape))
    print("te_x5.shape->{}".format(te_x5.shape))
    print("te_x6.shape->{}".format(te_x6.shape))
    #print("len(hist)->",len(hist[0]))
    x1_vocab = {}
    x2_vocab = {}
    x3_vocab = {}
    x4_vocab = {}
    x5_vocab = {}
    x6_vocab = {}
    for i in range(len(tr_x1[0])):
        x1_vocab["ID"+str(i)] = i
    for i in range(len(tr_x2[0])):
        x2_vocab["ID"+str(i)] = i
    for i in range(len(tr_x3[0])):
        x3_vocab["ID"+str(i)] = i
    for i in range(len(tr_x4[0])):
        x4_vocab["ID"+str(i)] = i
    for i in range(len(tr_x5[0])):
        x5_vocab["ID"+str(i)] = i
    for i in range(len(tr_x6[0])):
        x6_vocab["ID"+str(i)] = i
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
    x3_reverse_vocab = {x3_vocab[word]: word for word in x3_vocab}
    x3_indexed_vocab = [x3_reverse_vocab[index] for index in range(len(x3_reverse_vocab))]
    x4_reverse_vocab = {x4_vocab[word]: word for word in x4_vocab}
    x4_indexed_vocab = [x4_reverse_vocab[index] for index in range(len(x4_reverse_vocab))]
    x5_reverse_vocab = {x5_vocab[word]: word for word in x5_vocab}
    x5_indexed_vocab = [x5_reverse_vocab[index] for index in range(len(x5_reverse_vocab))]
    x6_reverse_vocab = {x6_vocab[word]: word for word in x6_vocab}
    x6_indexed_vocab = [x6_reverse_vocab[index] for index in range(len(x6_reverse_vocab))]
    ds_tr = TensorDataset(torch.from_numpy(tr_x1).float(),torch.from_numpy(tr_x2).float(),torch.from_numpy(tr_x3).float(),torch.from_numpy(tr_x4).float(), torch.from_numpy(tr_x5).float(),torch.from_numpy(tr_x6).float())
    ds_te = TensorDataset(torch.from_numpy(te_x1).float(),torch.from_numpy(te_x2).float(),torch.from_numpy(te_x3).float(),torch.from_numpy(te_x4).float(), torch.from_numpy(te_x5).float(),torch.from_numpy(te_x6).float())

    writer = SummaryWriter()
    model = MAVITM(
    topics = define_topic,
    joint_input = len(tr_x1[0]) + len(tr_x2[0]) + len(tr_x3[0]) + len(tr_x4[0]) + len(tr_x5[0]) + len(tr_x6[0]),
    input_x1 = len(tr_x1[0]),
    input_x2 = len(tr_x2[0]),
    input_x3 = len(tr_x3[0]),
    input_x4 = len(tr_x4[0]),
    input_x5 = len(tr_x5[0]),
    input_x6 = len(tr_x6[0]),
    hidden1_dimension = 100,
    hidden2_dimension = 100,
    )
    print(model)
    print('Training stage.')
    ae_optimizer = Adam(model.parameters(), 0.001, betas=(0.99, 0.999)) # 最適化アルゴリズム

    def training_callback(autoencoder, epoch, lr, loss, perplexity):
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'perplexity': perplexity,
        }, global_step=epoch)
        decoder_weight = model.x1_generator.linear.weight.detach().cpu()
        topics = [
            [x1_reverse_vocab[item.item()] for item in topic]
            for topic in decoder_weight.topk(top_words, dim=0)[1].t()
        ]
    train(
        ds_tr,
        define_topic,
        model,
        cuda=cuda,
        validation=ds_te,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=training_callback)
    #print('Evaluation stage.')
    """

    train_batch = 32
    trainloader = DataLoader(
        ds_train,
        batch_size=train_batch,
    )

    for i in range(1000):
        print(f"Epoch -> {i}")
        for x,t in enumerate(trainloader):
            #print(f"X1->{t[0]}")
            #print(f"X2->{t[0]}")
            mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge = model(t[0],t[1])
            #jmvae_zero_loss = model.jmvae_zero_loss(t[0], t[1], mean, logvar, x1_recon, x1_mean, x1_logvar, x2_recon, x2_mean, x2_logvar).mean()
            #kl_x1_x2 = model.kl_x1_x2(mean, logvar, x1_mean, x1_logvar, x2_mean, x2_logvar).mean()
            #print(f'jmvae_zero_loss -> {jmvae_zero_loss}')
            #print(f'kl_x1_x2 -> {kl_x1_x2}')
            #loss = model.jmvae_kl_loss(jmvae_zero_loss, kl_x1_x2, 10)
            loss = model.telbo(t[0], t[1], mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, 1.0, 1.0, 1.0).mean()

            print(f'loss -> {loss}')

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step(closure=None)
    """
if __name__ == '__main__':
    main()
