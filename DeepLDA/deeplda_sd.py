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
    define_topic = 3 # トピックの数を事前に定義
    hist = np.loadtxt( "generateSD/hist.txt" , dtype=np.int32)
    hist = np.array(hist,dtype=float)
    """
    データセットの読み込み
    BoFヒストグラムの作成
    """
    print("作成したヒストグラム->\n"+str(hist))
    print(hist.shape)
    print("len(hist)->",len(hist[0]))
    vocab = {}
    for i in range(len(hist[0])):
        vocab["ID"+str(i)] = i
    print("vocab->",vocab)
    label = np.loadtxt( "generateSD/label.txt" , dtype=np.int32)
    print(label)
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
        """
        topics:各トピックの上位単語をリストで格納
        """
        print("topics->"+str(topics))
        """
        １訓練訓練終了後に各トピックの単語をテキストファイルに保存
        """
        file_name = "./topic.txt"
        file = open(file_name, 'w')
        for index, topic in enumerate(topics): # 各トピック(50)を印字してファイルに保存
            print(str(index)+"番目のトピック" + ':' + ','.join(topic))
            file.write(str(index) + ':' + ','.join(topic) + "\n")
        file.close()
    #################################################################################
    ds_train = TensorDataset(torch.from_numpy(hist).float())
    ds_val = TensorDataset(torch.from_numpy(hist).float())
    autoencoder = ProdLDA(
        in_dimension=len(hist[0]),# 本来はlen(vocab),1995,ただし,ヒストグラムの次元数と等しい
        hidden1_dimension=100,
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
    #print(topics)
    for topic in topics:
        print(','.join(topic))
        #file = open(file_name, 'w')
        #file.write(','.join(topic))
    if not testing_mode:
        writer.add_embedding(
            autoencoder.encoder.linear1.weight.detach().cpu().t(),
            metadata=indexed_vocab,
            tag='feature_embeddings',
        )

    writer.close()
    """
    評価
    """
    test_batch = 100
    dataloader = DataLoader(
        ds_train,
        batch_size=test_batch,
    )
    label = [] # 仮のラベル
    for i in range(test_batch):
        label.append(i)
    label = np.array(label)
    #print(label)
    print("decoder_weight->\n"+str(decoder_weight.t()))
    print("decoder_weight.shape->\n"+str(decoder_weight.t().shape))
    print("decoder_weight.topk->\n"+str(decoder_weight.topk(20, dim=0)[1].t()))
    autoencoder.eval()
    from torch.autograd import Variable
    # 潜在変数の可視化
    from sklearn.manifold import TSNE
    from random import random
    import matplotlib.pyplot as plt

    # colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
    def visualize_zs(zs, labels):
        plt.figure(figsize=(10,10))
        points = TSNE(n_components=2, random_state=0).fit_transform(zs)
        for p, l in zip(points, labels):
            plt.scatter(p[0], p[1], marker="${}$".format(l))
        plt.savefig('figure.png')
        #plt.show()

    #print("autoencoder->",autoencoder)
    for x,t in enumerate(dataloader):
        print(x,t)
        #print("x->",t[0])
        #print("autoencoder(t[0])->",autoencoder.encode(t[0]))
        #print("autoencoder(t[0])->",autoencoder(t[0]))
        #a, b, c = autoencoder.encode(Variable(t[0], volatile=True))
        a, b, c, z = autoencoder(t[0])
        z = z.cpu()
        #print("a.shape->",a.shape)
        #print("b->",b)
        #print("c->",c)
        #print("y.shape->"+str(y.shape))
        #print("len(y)->",str(len(y)))
        #print("y->"+str(y))

        #print("model(x)->",model(x))
        #print("z.shape->"+str(z.shape))
        #print("len(z)->",str(len(z)))
        #print("z->"+str(z))
        #print(autoencoder)

        visualize_zs(z.detach().numpy(), label)
        break

if __name__ == '__main__':
    main()
