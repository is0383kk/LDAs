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
# BoFヒストグラム作成用のmodule
from module.bow import make_bof
from module.bow import make_codebook
# DeepLDA用の訓練用関数とvaeモデル
from ptavitm.model import train
from ptavitm.vae import ProdLDA
# データローダ
from torch.utils.data import DataLoader


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
    help='バッチサイズ (default 200).',
    type=int,
    default=32
)
@click.option(
    '--epochs',
    help='学習エポック (default 5).',
    type=int,
    default=50
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
def main(cuda,batch_size,epochs,top_words,testing_mode):#上のコマンドライン引数
    t1 = time.time() # 処理前の時刻
    define_topic = 3 # トピックの数を事前に定義
    hist = np.loadtxt( "/home/yoshiwo/workspace/res/study/make_synthetic_data/hist.txt" , dtype=float)
    label = np.loadtxt( "/home/yoshiwo/workspace/res/study/make_synthetic_data/label.txt" , dtype=np.int32)
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    print("作成したヒストグラム->\n"+str(hist))
    print("hist.shape->{}".format(hist.shape))
    print("len(hist)->",len(hist[0]))
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
    ds_val = TensorDataset(torch.from_numpy(hist).float(),torch.from_numpy(label).int())
    autoencoder = ProdLDA(
        in_dimension=len(hist[0]),# 入力,本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
        hidden1_dimension=50, # 中間層
        hidden2_dimension=50,
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
    t2 = time.time()
    # 経過時間を表示
    elapsed_time = t2-t1
    print(f"経過時間：{elapsed_time}")

    ##################メイン処理はここまで########################################################
    """
    各文書の潜在変数を可視化してクラスタリング
    """
    test_batch = 100
    dataloader = DataLoader(
        ds_train,
        batch_size=test_batch,
    )


    #print("decoder_weight->\n"+str(decoder_weight.t()))
    #print("decoder_weight.shape->\n"+str(decoder_weight.t().shape))
    #print("decoder_weight.topk->\n"+str(decoder_weight.topk(top_words, dim=0)[1].t()))

    autoencoder.eval()

    # 潜在変数の可視化
    from sklearn.manifold import TSNE
    from random import random
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    #prior_mean = torch.full((200,3),0.0000)
    #prior_logvar = torch.full((200,3),-0.4055)
    #eps = prior_mean.new().resize_as_(prior_mean).normal_(mean=0, std=1)
    #prior_z = prior_mean + prior_logvar.exp().sqrt() * eps
    #prior_z = prior_z.cpu()
    #print("prior_z-.{}".format(prior_z))
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

    def visualize_zs(zs, labels):
        #plt.figure(figsize=(10,10))
        fig = plt.figure(figsize=(10,10))
        #ax = Axes3D(fig)
        points = TSNE(n_components=2, random_state=0).fit_transform(zs)
        for p, l in zip(points, labels):
            plt.title("Latent space (Topic:"+str(define_topic)+", Doc:"+str(hist.shape[0])+", Words:"+str(hist.shape[1])+")", fontsize=24)
            plt.xlabel("Latent space:xlabel", fontsize=21)
            plt.ylabel("Latent space:ylabel", fontsize=21)
            plt.scatter(p[0], p[1], marker="${}$".format(l),c=colors[l],s=100)
            #ax.scatter(p[0], p[1], p[2], marker="${}$".format(l),c=colors[l])
        plt.savefig('./sample_z/'+'k'+str(define_topic)+'v'+str(hist.shape[1])+'d'+str(hist.shape[0])+'.png')
        #plt.savefig('document_z3d.png')
        #plt.show()

    for x,t in enumerate(dataloader):
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
        recon, mean, logvar, z = autoencoder(t[0]) # 訓練後の潜在変数の抽出

        z = z.cpu()
        z_label = t[1].cpu()

        #z2_label = t[1].cpu()
        #print("ARI->",ar(z_label.numpy(),z2_label.numpy()))

        #print("autoencoder.decode(mean,logvar)->{}".format(autoencoder.decode(mean,logvar))) # batch x １文書中の単語数
        #print("autoencoder.decode(mean,logvar)->{}".format(autoencoder.decode(mean,logvar).shape))
        #print("mean->",mean)
        #print("logvar->",logvar)
        """
        潜在変数zの確認
        """
        #print("z.shape->"+str(z.shape))
        #print("len(z)->",str(len(z)))
        #print("z->"+str(z))

        #visualize_zs(prior_z.detach().numpy(), z_label.cpu().detach().numpy())
        visualize_zs(z.detach().numpy(), z_label.cpu().detach().numpy())
        break

if __name__ == '__main__':
    main()
