import os
import sys
import codecs
try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#ROSが干渉してくる
import cv2
import click
import numpy as np


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


def make_bof( code_book_name, images, hist_name ): # BoF用->calc_feature関数,make_codebook関数を使用
    code_book = np.loadtxt( code_book_name, dtype=np.float32 )

    knn= cv2.ml.KNearest_create()
    knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book),dtype=np.float32))
    hist = []
    for img in images:
        f = calc_feature( img )
        idx = knn.findNearest( f, 1 )[1]
        h = np.zeros( len(code_book) )
        for i in idx:
            h[int(i)] += 1
        hist.append( h.tolist() )
    return hist

def make_bow(src_name): # BoW用->これ単体のみを使用
    word_dic = []
    vocab = {}
    # 各行を単語に分割
    lines = []
    for line in codecs.open( src_name, "r", "utf8" ).readlines():
        # 改行コードを削除
        line = line.rstrip("\r\n")

        # 単語分割
        words = line.split(" ")

        lines.append( words )
    print("lines(" + str(len(lines)) + ")->" + str(lines))

    # 単語辞書とヒストグラムを作成
    i = 0
    for words in lines:
        for w in words:
            # 単語がなければ辞書に追加
            if not w in word_dic:
                word_dic.append( w )
                vocab[w] = i
                i = i + 1
    print("vacab("+str(len(vocab))+")->"+str(vocab))
    print("word_dic("+ str(len(word_dic))+ ")->" + str(word_dic))
    print("BoWヒストグラムの作成を行います")

    hist = np.zeros( (len(lines), len(word_dic)) )

    print("\n")
    for d,words in enumerate(lines):
        for w in words:
            idx = word_dic.index(w)
            hist[d,idx] += 1
    return vocab,hist
