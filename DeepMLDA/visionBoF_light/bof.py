import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')#ROSが干渉してくる
import cv2
import numpy as np
import glob

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

    hists = []
    for img in images:
        f = calc_feature( img )
        idx = knn.findNearest( f, 1 )[1]

        h = np.zeros( len(code_book) ,dtype=np.int64)
        
        for i in idx:
            h[int(i)] += 1

        hists.append( h.tolist() )
    
    
    print(hists)

    np.savetxt( hist_name, hists, fmt=str("%d")  )


def main():
    files = glob.glob("images/*.png")
    make_codebook( files, 50, "codebook.txt" )
    make_bof( "codebook.txt", files, "histogram_v.txt" )

if __name__ == '__main__':
    main()
