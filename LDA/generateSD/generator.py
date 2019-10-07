## LDAの生成過程を用いたSynthetic dataの生成を行うプログラム
import math
import random
import numpy
import numpy.random
import sys

# ハイパーパラメータの定義
TOPIC_N = 3 # トピック数
VOCABULARY_SIZE = 150 # 単語数
DOC_NUM = 1000 # 文書数
TERM_PER_DOC = 200 # ドキュメントごとの単語数

beta = [0.01 for i in range(VOCABULARY_SIZE)] # ディレクレ分布のパラメータ(グラフィカルモデル左端)
alpha = [0.9 for i in range(TOPIC_N)] # #ディレクレ分布のパラメータ(グラフィカルモデル右端)
#FILE_NAME = sys.argv[1] # 保存先のファイル名
FILE_NAME = "synthetic_data" # 保存先のファイル名


hist = numpy.zeros( (DOC_NUM, TERM_PER_DOC) ) # ヒストグラム格納用の変数

#print(hist)
#print(hist[0])

phi = []
# generate multinomial distribution over words for each topic
# 各トピックの単語にわたる多項分布を生成
for i in range(TOPIC_N):
    topic =	numpy.random.mtrand.dirichlet(beta, size = 1)
    phi.append(topic)

output_f = open(FILE_NAME+'.doc','w')
z_f = open(FILE_NAME+'.z','w')
theta_f = open(FILE_NAME+'.theta','w')

hist_i = 0 # ヒストグラムの縦の要素のインデックス
# 各ドキュメントの単語を生成
for i in range(DOC_NUM):
    buffer = {}
    z_buffer = {} # 真のzをトラッキングするための変数
    # θのサンプリング
    theta = numpy.random.mtrand.dirichlet(alpha,size = 1)
    for j in range(TERM_PER_DOC):
        # zのサンプリング
        z = numpy.random.multinomial(1,theta[0],size = 1)
        z_assignment = 0
        for k in range(TOPIC_N):
            if z[0][k] == 1:
            	break
            z_assignment += 1
        if not z_assignment in z_buffer:
            z_buffer[z_assignment] = 0
        z_buffer[z_assignment] = z_buffer[z_assignment] + 1
        # トピックzからサンプリングされる観測w
        w = numpy.random.multinomial(1,phi[z_assignment][0],size = 1)
        w_assignment = 0
        for k in range(VOCABULARY_SIZE):
            if w[0][k] == 1:
                break
            w_assignment += 1
        if not w_assignment in buffer:
            buffer[w_assignment] = 0
        buffer[w_assignment] = buffer[w_assignment] + 1

    # output
    output_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
    for word_id, word_count in buffer.items():
        output_f.write(str(word_id)+':' + str(word_count)+' ')
        hist[hist_i,word_id] = word_count
    output_f.write('\n')
    z_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
    for z_id, z_count in z_buffer.items():
        z_f.write(str(z_id)+':'+str(z_count)+' ')
    z_f.write('\n')
    theta_f.write(str(i)+'\t')
    for k in range(TOPIC_N):
        theta_f.write(str(k)+':'+str(theta[0][k])+' ')
    theta_f.write('\n')
    hist_i += 1
z_f.close()
theta_f.close()
output_f.close()

# phiを格納
output_f = open(FILE_NAME+'.phi','w')
for i in range(TOPIC_N):
    output_f.write(str(i)+'\t')
    for j in range(VOCABULARY_SIZE):
        output_f.write(str(j)+':'+str(phi[i][0][j])+' ')
    output_f.write('\n')
output_f.close()

# ハイパーパラメータを格納
output_f = open(FILE_NAME+'.hyper','w')
output_f.write('TOPIC_N:'+str(TOPIC_N)+'\n')
output_f.write('VOCABULARY_SIZE:'+str(VOCABULARY_SIZE)+'\n')
output_f.write('DOC_NUM:'+str(DOC_NUM)+'\n')
output_f.write('TERM_PER_DOC:'+str(TERM_PER_DOC)+'\n')
output_f.write('alpha:'+str(alpha[0])+'\n')
output_f.write('beta:'+str(beta[0])+'\n')
output_f.close()
print(hist)
numpy.savetxt( "hist.txt", hist, fmt=str("%d") )
