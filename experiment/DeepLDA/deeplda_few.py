import click
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import pickle
# BoWヒストグラム作成用のmodule
from module.bow import make_bow
# DeepLDA用の訓練用関数とvaeモデル
from ptavitm.model import train
from ptavitm.vae import ProdLDA



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
    help='バッチサイズ.',
    type=int,
    default=2
)
@click.option(
    '--epochs',
    help='学習エポック.',
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
    sentence_file = "./few_input_data/text.txt"
    hist_file = "./few_input_data/hist.txt"
    word_dic = "./few_input_data/word_dic.txt"
    hist_k = 10 # ヒストグラムの水増し係数
    """
    データセットの読み込み
    BoWヒストグラムの作成
    """
    vocab, hist = make_bow(sentence_file)
    print("hist->",hist)
    # ヒストグラム化
    hist = hist * hist_k
    print("作成したヒストグラム->"+str(hist))
    print("len(hist)->",len(hist[0]))
    #######################ここまでがBoWを作成する作業##############################################
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
