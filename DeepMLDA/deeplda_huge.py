# CSV読み込み関数用
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import textacy
# メイン処理部分用
import click
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus
import torch
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from scipy.sparse import load_npz
from tensorboardX import SummaryWriter
import pickle
# ptavitmフォルダ用
from ptavitm.model import train
from ptavitm.vae import ProdLDA
from ptavitm.utils import CountTensorDataset


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
    default=200
)
@click.option(
    '--epochs',
    help='学習エポック.',
    type=int,
    default=10
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
@click.option(
    '--verbose-mode',
    help='whether to run in verbose mode (default False).',
    type=bool,
    default=False
)



# メイン処理
def main(
    cuda,
    batch_size,
    epochs,
    top_words,
    testing_mode,
    verbose_mode
):
    print('Loading input data')
    # TODO fix relative paths
    data_train = load_npz('npz_data/train.txt.npz')
    data_val = load_npz('npz_data/test.txt.npz')
    corpus = Sparse2Corpus(data_train, documents_columns=False)
    with open('npz_data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    reverse_vocab = {vocab[word]: word for word in vocab}
    indexed_vocab = [reverse_vocab[index] for index in range(len(reverse_vocab))]
    writer = SummaryWriter()  # create the TensorBoard object

    # callback function to call during training, uses writer from the scope
    def training_callback(autoencoder, epoch, lr, loss, perplexity):
        if verbose_mode:
            decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
            topics = [
                [reverse_vocab[item.item()] for item in topic]
                for topic in decoder_weight.topk(top_words, dim=0)[1].t()
            ]
            cm = CoherenceModel(
                topics=topics,
                corpus=corpus,
                dictionary=Dictionary.from_corpus(corpus, reverse_vocab),
                coherence='u_mass'
            )
            coherence = cm.get_coherence()
            coherences = cm.get_coherence_per_topic()
            file_name = "./topic.txt"
            file = open(file_name, 'w')
            for index, topic in enumerate(topics):
                print(str(index) + ':' + str(coherences[index]) + ':' + ','.join(topic))
                file.write(str(index) + ':' + ','.join(topic) + "\n")
            print(coherence)
            file.close()
        else:
            coherence = 0
        writer.add_scalars('data/autoencoder', {
            'lr': lr,
            'loss': loss,
            'perplexity': perplexity,
            'coherence': coherence,
        }, global_step=epoch)
    # ここからCSV読み込み
    # ここまで
    ds_train = CountTensorDataset(data_train)
    ds_val = CountTensorDataset(data_val)
    autoencoder = ProdLDA(
        in_dimension=len(vocab),
        hidden1_dimension=100,
        hidden2_dimension=100,
        topics=5
    )
    if cuda:
        autoencoder.cuda()
    print('Training stage.')
    ae_optimizer = Adam(autoencoder.parameters(), 0.0001, betas=(0.99, 0.999))
    train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        update_callback=training_callback,
        sampler=WeightedRandomSampler(torch.ones(data_train.shape[0]), 20000),
        num_workers=4
    )
    autoencoder.eval()
    decoder_weight = autoencoder.decoder.linear.weight.detach().cpu()
    topics = [
        [reverse_vocab[item.item()] for item in topic]
        for topic in decoder_weight.topk(top_words, dim=0)[1].t()
    ]
    cm = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=Dictionary.from_corpus(corpus, reverse_vocab),
        coherence='u_mass'
    )
    coherence = cm.get_coherence()
    coherences = cm.get_coherence_per_topic()
    for index, topic in enumerate(topics):
        print(str(index) + ':' + str(coherences[index]) + ':' + ','.join(topic))
    print(coherence)
    if not testing_mode:
        writer.add_embedding(
            autoencoder.encoder.linear1.weight.detach().cpu().t(),
            metadata=indexed_vocab,
            tag='feature_embeddings',
        )
    #torch.save(autoencoder.state_dict(),'./weight/deepLDA.pth')
    writer.close()



    dataloader = DataLoader(
        ds_train,
        batch_size=5,
    )

    autoencoder.eval()
    from torch.autograd import Variable
    print("autoencoder->",autoencoder)
    for x,t in enumerate(dataloader):
        #print("x->",t[0])
        print("autoencoder(t[0])->",autoencoder.encode(t[0]))
        #a, b, c = autoencoder.encode(Variable(t[0], volatile=True))
        a, b, c = autoencoder.encode(t[0])
        print("a->",a)
        print("a.shape->",a.shape)
        print("b->",b)
        print("c->",c)
        #print("y.shape->"+str(y.shape))
        #print("len(y)->",str(len(y)))
        #print("y->"+str(y))

        #print("model(x)->",model(x))
        #print("z.shape->"+str(z.shape))
        #print("len(z)->",str(len(z)))
        #print("z->"+str(z))
        #print(autoencoder)



if __name__ == '__main__':
    main()
