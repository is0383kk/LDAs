import codecs
import numpy as np

def make_bow( src_name, hist_name, dict_name ):
    word_dic = []

    # 各行を単語に分割
    lines = []
    """
    src_nameファイルを開く

    """
    for line in codecs.open( src_name, "r", "utf8" ).readlines():
        # 改行コードを削除
        line = line.rstrip("\r\n")

        # 単語分割
        words = line.split(" ")

        lines.append( words )
    """
    ファイル内が
    吉田 首相 が 国会 で 演説
    吉田 首相 が 試合 を 観戦
    とすると
    lines = [['吉田','首相','が','国会','で','演説'],
            ['吉田','首相','が','試合','を','観戦']]
            となる
    """
    print("lines(" + str(len(lines)) + ")->" + str(lines))

    # 単語辞書とヒストグラムを作成
    for words in lines:
        for w in words:
            # 単語がなければ辞書に追加
            if not w in word_dic:
                word_dic.append( w )
    """
    word_dic = ['吉田','首相','が']
    のような形で単語をリストに突っ込む
    """
    print("word_dic("+ str(len(word_dic))+ ")->" + str(word_dic))

    # ヒストグラム化
    hist = np.zeros( (len(lines), len(word_dic)) )
    print("\n")
    for d,words in enumerate(lines):
        for w in words:
            idx = word_dic.index(w)
            hist[d,idx] += 1


    np.savetxt( hist_name, hist, fmt=str("%d") )
    codecs.open( dict_name, "w", "utf8" ).write( "\n".join( word_dic ) )

    """
    [['吉田', 'を'], ['吉田', 'は']]
    ['吉田', 'を', 'は']

    hist = [[1. 1. 0.]
           [1. 0. 1.]]
    """

    print(hist)

def main():
    make_bow( "text.txt", "histogram_w.txt", "word_dic.txt" )

if __name__ == '__main__':
    main()
