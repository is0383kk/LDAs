import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#ROSが干渉してくる
import cv2
import glob
import click
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import pickle

from ptavitm.model import train
from ptavitm.vae import ProdLDA
import codecs
from ptavitm.model import train
from ptavitm.vae import ProdLDA

_detecotor = cv2.AKAZE_create()

def calc_feature( filename ):
    img = cv2.imread( filename, 0 )
    kp, discriptor = _detecotor.detectAndCompute(img,None)
    return np.array(discriptor, dtype=np.float32 )

# コードブックを作成
def make_codebook( images, code_book_size, save_name ):
    bow_trainer = cv2.BOWKMeansTrainer( code_book_size )
    for img in images:
        f = calc_feature(img)  # 特徴量計算
        bow_trainer.add( f )
    code_book = bow_trainer.cluster()
    np.savetxt( save_name, code_book )

# ヒストグラム作成
def make_bof( code_book_name, images, hist_name ):
    code_book = np.loadtxt( code_book_name, dtype=np.float32 )

    knn= cv2.ml.KNearest_create()
    knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book),dtype=np.float32))
    hist = []
    for img in images:
        f = calc_feature( img )
        idx = knn.findNearest( f, 1 )[1]
        h = np.zeros( len(code_book) ,dtype=np.int64)
        for i in idx:
            h[int(i)] += 1
        hist.append( h.tolist() )
    return hist

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
    default=2
)
@click.option(
    '--epochs',
    help='学習エポック (default 5).',
    type=int,
    default=5
)
@click.option(
    '--top-words',
    help='各トピックにおいて表示するトップ単語の数 (default 12).',
    type=int,
    default=12
)
@click.option(
    '--testing-mode',
    help='テストモードで実行するかどうか (default False).',
    type=bool,
    default=False
)
def main(cuda,batch_size,epochs,top_words,testing_mode):#上のコマンドライン引数
    define_topic = 3
    image_feature = "番目の特徴"
    image_file = "./visionBoF_light/images/*.png"
    codebook_file = "./visionBoF_light/codebook.txt"
    hist_file = "./visionBoF_light/histgram_v.txt"
    hist_k = 10 # ヒストグラムの水増し係数
    """
    データセットの読み込み
    """
    files = glob.glob(image_file)
    make_codebook( files, 50, codebook_file )
    hist = make_bof( codebook_file , files, hist_file )
    # ヒストグラム化
    hist = hist * hist_k
    print("作成したヒストグラム->"+str(hist))
    print("len(hist)->",len(hist[0]))
    vocab = []
    """
    vocab
    {'0番目の特徴': 0,'1番目の特徴':1 } 
    """

    #################################################################################
    print('Loading input data')
    reverse_vocab = {vocab[word]: word for word in vocab};
    indexed_vocab = [reverse_vocab[index] for index in range(len(reverse_vocab))]
#########################################################################################################################
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
        in_dimension=len(hist[0]),# len(vocab),1995
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


if __name__ == '__main__':
    main()
