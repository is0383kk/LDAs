## LDAの生成過程を用いたSynthetic dataの生成を行うプログラム
import math
import random
import numpy as np
import sys
import click

@click.command()
@click.option('--topic_n', help = 'トピック数', type=int, default = 5)
@click.option('--vacabulary_size', help = '単語数', type=int, default = 50)
@click.option('--doc_num', help = '文書数（ヒストグラムの列数）', type=int, default = 3000)
@click.option('--term_per_doc', help = '文書ごとの単語数（ヒストグラムの行数）', type=int, default = 50)
@click.option('--mode', help = 'zを固定するかどうか(Falseで固定,Trueで固定しない)', type=bool, default = False)
@click.option('--test', help = 'テスト用のデータ作成(Falseで訓練用,Trueでテスト用)', type=bool, default = False)

def main(topic_n,
	vacabulary_size,
	doc_num,
	term_per_doc,
	mode,
	test
	):
	if test == True:
	    doc_num = 1000
	    
	# ハイパーパラメータの定義
	TOPIC_N = topic_n # トピック数
	VOCABULARY_SIZE = vacabulary_size # 単語数
	DOC_NUM = doc_num # 文書数
	TERM_PER_DOC = term_per_doc # ドキュメントごとの単語数
	MODE = mode

	beta = [1.0 for i in range(VOCABULARY_SIZE)] # ディレクレ分布のパラメータ(グラフィカルモデル右端)
	alpha = [1.0 for i in range(TOPIC_N)] # #ディレクレ分布のパラメータ(グラフィカルモデル左端)


	#print("alpha->",alpha)
	#print("beta->",beta)
	#FILE_NAME = sys.argv[1] # 保存先のファイル名
	FILE_NAME = "synthetic_data" # 保存先のファイル名



	hist = np.zeros( (DOC_NUM, TERM_PER_DOC) ) # ヒストグラム格納用の変数
	document_label = np.zeros(DOC_NUM) # 単語の潜在変数を元に文書ラベルを決定する変数
	z_max = -1145141919810 # z_countと比較するための変数

	#print(document_label[0])
	#print(hist)
	#print(hist[0])

	# 各トピックの単語にわたる多項分布を生成
	phi = []
	topic = []
	for i in range(TOPIC_N):
		if (MODE == False):
			beta = [0.1 for i in range(VOCABULARY_SIZE)]
			#beta[0] = 10
			topic = np.random.mtrand.dirichlet(beta, size = 1)
			#print("topic->{}".format(topic))
		else:
			topic = np.random.mtrand.dirichlet(beta, size = 1)
			#print("topic->{}".format(topic))

		phi.append(topic)

	#print("phi->",phi)
	# 各ファイル変数
	output_f = open(FILE_NAME+'.doc','w')
	z_f = open(FILE_NAME+'.z_feature','w')
	theta_f = open(FILE_NAME+'.theta','w')


	"""
	ここから生成
	"""

	"""
	hoge = np.random.mtrand.dirichlet(alpha,size = none)について
	alpheに依存したディリクレ分布の乱数を生成：
		生成される乱数は合計して 1 になる配列
		これがsize分だけ生成される

	hoge = np.random.multinomial(n, pvals, size=None)について
	pvalsに依存した多項分布の生成：
		pvals　は合計して 1である配列
		pvals = [1.0, 0, 0]
		これを確率として考えると
		hoge = [n,0,0]がsize分だけ生成される
		zの生成で使用

	"""
	hist_i = 0 # ヒストグラムの縦の要素のインデックス
	remove_label = [] # 潜在ラベルの重複インデックスを格納
	# 各ドキュメントの単語を生成
	for i in range(DOC_NUM):
		print("epochs->{}".format(i))
		buffer = {}
		z_buffer = {} # 真のzをトラッキングするための変数
		theta = np.zeros((1,TOPIC_N), dtype = float)
		# θのサンプリング
		if (MODE == True):
			theta[0][i%TOPIC_N] = 1.0
		else:
			theta = np.random.mtrand.dirichlet(alpha,size = 1)

		for j in range(TERM_PER_DOC):
			# zのサンプリング
			z = np.random.multinomial(1,theta[0],size = 1)
			z_assignment = 0
			for k in range(TOPIC_N):
				if z[0][k] == 1:
					break
				z_assignment += 1
			if not z_assignment in z_buffer:
				z_buffer[z_assignment] = 0
			z_buffer[z_assignment] = z_buffer[z_assignment] + 1
			# トピックzからサンプリングされる観測w

			w = np.random.multinomial(1,phi[z_assignment][0],size = 1)
			w_assignment = 0
			for k in range(VOCABULARY_SIZE):
				if w[0][k] == 1:
					break
				w_assignment += 1
			if not w_assignment in buffer:
				buffer[w_assignment] = 0
			buffer[w_assignment] = buffer[w_assignment] + 1


		#print("buffer->",buffer)
		#print("theta->",theta)
		#print("phi->",phi)
		#print("EPOCH={}----------------------".format(i))
		#print("z->", z)
		#print("z_assignment->", z_assignment)
		#print("z_buffer->", z_buffer)
		#print("----------------------")
		#print("w->", w)
		#print("w_assignment->", w_assignment)
		#print("buffer->", buffer)

		"""
		ここまで人口データ作成
		"""
		# ここから各情報をファイルに保存
		output_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
		for word_id, word_count in buffer.items():
			output_f.write(str(word_id)+':' + str(word_count) + ' ')
			#output_f.write("単語(" + str(word_id)+ ")" +':' + str(word_count)+'個, ')
			hist[hist_i,word_id] = word_count
		output_f.write('\n')
		z_f.write(str(i)+'\t'+str(TERM_PER_DOC)+'\t')
		for z_id, z_count in z_buffer.items():
			if (z_max == z_count):
				remove_label.append(hist_i)

			if (z_max < z_count): # z_countが最大の時のz_idを文書ラベルとして採用する
				z_max = z_count
				#print("z_max,z_count",z_max,z_count)
				document_label[hist_i] = z_id

			z_f.write( str(z_id)  + ':'+str(z_count)+' ')
		z_f.write('\n')
		theta_f.write(str(i)+'\t')
		for k in range(TOPIC_N):
			theta_f.write(str(k)+':'+str(theta[0][k])+' ')
		theta_f.write('\n')

		hist_i += 1 # ヒストグラムの縦のインデックス
		z_max = -114514 # z_countと比較する最大値の初期化

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
	#print("label->{}".format(len(document_label)))
	#print("label->{}".format(document_label))
	print("\nremove_label->",remove_label)
	if len(remove_label) != 0:
		hist = np.delete(hist,remove_label,0)
		document_label = np.delete(document_label,remove_label,0)
	#print("hist->{}".format(len(document_label)))
	#print("hist->{}".format(document_label))
	#print("label->{}".format(document_label))
	#print(document_label)
	if test == True:
		np.savetxt( "test_hist.txt", hist, fmt=str("%d") )
		np.savetxt( "test_label.txt", document_label, fmt=str("%d") )
	else:
		np.savetxt( "hist.txt", hist, fmt=str("%d") )
		np.savetxt( "label.txt", document_label, fmt=str("%d") )
	print("モード選択->{}".format(MODE))
	print("データ選択->{}".format(test))

if __name__ == "__main__":
	main()
