"""
LDAの生成過程を用いたSynthetic dataの生成を行うプログラム
LDAの推論用の人工データを生成
"""
import math
import random
import numpy as np
import sys
import click

@click.command()
@click.option('--topic_n', help = 'トピック数', type=int, default = 3)
@click.option('--vacabulary_size', help = '単語数', type=int, default = 12)
@click.option('--doc_num', help = '文書数（ヒストグラムの列数）', type=int, default = 30)
@click.option('--term_per_doc', help = '文書ごとの単語数（ヒストグラムの行数）', type=int, default = 12)
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
	    doc_num = 10
	    
	# ハイパーパラメータの定義
	TOPIC_N = topic_n # トピック数
	VOCABULARY_SIZE = vacabulary_size # 単語数
	DOC_NUM = doc_num # 文書数
	TERM_PER_DOC = term_per_doc # ドキュメントごとの単語数
	MODE = mode

	beta = [0.1 for i in range(VOCABULARY_SIZE)] # ディレクレ分布のパラメータ(グラフィカルモデル右端)
	alpha = [0.1 for i in range(TOPIC_N)] # #ディレクレ分布のパラメータ(グラフィカルモデル左端)

	FILE_NAME = "synthetic_data" # 保存先のファイル名


	hist = np.zeros( (DOC_NUM, TERM_PER_DOC) ) # ヒストグラム格納用の変数
	document_label = np.zeros(DOC_NUM) # 単語の潜在変数を元に文書ラベルを決定する変数
	z_max = -1145141919810 # z_countと比較するための変数
	tpd_t = int(TERM_PER_DOC / TOPIC_N) # 不正操作用の変数，単語生成確率を操作
	prob = float(1 / tpd_t) # 不正操作用の変数

	# 各トピックの単語にわたる多項分布を生成
	phi = []
	topic = []
	for i in range(TOPIC_N):
		if MODE == True: # 不正操作モード
			topic = np.zeros(TERM_PER_DOC)
			topic[i*tpd_t : tpd_t*(i+1)] = prob
			phi.append(topic)
		else: # 一様乱数から生成
			topic = np.random.mtrand.dirichlet(beta, size = 1)
			phi.append(topic)

	hist_i = 0 # ヒストグラムの縦の要素のインデックス
	remove_label = [] # 潜在ラベルの重複インデックスを格納
	# 各ドキュメントの単語を生成
	for i in range(DOC_NUM):
		print("epochs->{}".format(i))
		buffer = {}
		z_buffer = {} # 真のzをトラッキングするための変数
		theta = np.zeros((1,TOPIC_N), dtype = float)
		# θのサンプリング(トピック割当て確率を示す)
		if (MODE == True):
			theta[0][i%TOPIC_N] = 1.0
		else:
			theta = np.random.mtrand.dirichlet(alpha,size = 1)
		#print(f"theta -> {theta}")
		for j in range(TERM_PER_DOC):
			# zのサンプリング（生成されるトピック）
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
			if MODE == True:
				# 不正操作モード
				w = np.random.multinomial(1,phi[z_assignment],size = 1)
			else:
				# 一様乱数モード
				w = np.random.multinomial(1,phi[z_assignment][0],size = 1)
			
			w_assignment = 0
			for k in range(VOCABULARY_SIZE):
				if w[0][k] == 1:
					break
				w_assignment += 1
			if not w_assignment in buffer:
				buffer[w_assignment] = 0
			buffer[w_assignment] = buffer[w_assignment] + 1

		print("------------------------------------------")

		"""
		ここまで人口データ作成
		"""
		for word_id, word_count in buffer.items():
			hist[hist_i,word_id] = word_count
			
		for z_id, z_count in z_buffer.items():
			if (z_max == z_count):
				remove_label.append(hist_i)

			if (z_max < z_count): # z_countが最大の時のz_idを文書ラベルとして採用する
				z_max = z_count
				document_label[hist_i] = z_id

		hist_i += 1 # ヒストグラムの縦のインデックス
		z_max = -114514 # z_countと比較する最大値の初期化
	
	if len(remove_label) != 0:
		hist = np.delete(hist,remove_label,0)
		document_label = np.delete(document_label,remove_label,0)
	
	if test == False:
		np.savetxt( "k"+str(topic_n)+"tr.txt", hist, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"tr_z.txt", document_label, fmt=str("%d") )
	else:
		np.savetxt( "k"+str(topic_n)+"te.txt", hist, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"te_z.txt", document_label, fmt=str("%d") )
	print("トピックが重複している文書",remove_label)
	print("不正操作モード選択->{}".format(MODE))
	print("テストデータ生成->{}".format(test))

if __name__ == "__main__":
	main()
