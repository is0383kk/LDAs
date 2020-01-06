## LDAの生成過程を用いたSynthetic dataの生成を行うプログラム
"""
基本的に不正操作(theta, phi_w, phi_f を手動設定)しないとARIが0に近い値を算出する
"""
import math
import random
import numpy as np
import sys
import click

@click.command()
@click.option('--topic_n', help = 'トピック数', type=int, default = 30)
@click.option('--vacabulary_size', help = '単語数', type=int, default = 360)
@click.option('--hist_num', help = '文書数（ヒストグラムの列数）', type=int, default = 1000)
@click.option('--term_per_doc', help = '文書ごとの単語数（ヒストグラムの行数）', type=int, default = 360)
@click.option('--mode', help = 'zを固定するかどうか(Falseで固定,Trueで固定しない)', type=bool, default = False)
@click.option('--test', help = 'テスト用のデータ作成(Falseで訓練用,Trueでテスト用)', type=bool, default = False)

def main(topic_n,
	vacabulary_size,
	hist_num,
	term_per_doc,
	mode,
	test):
	if test == True:
	    hist_num = 1000

	# ハイパーパラメータの定義
	TOPIC_N = topic_n # トピック数
	HIST_W_SIZE	 = vacabulary_size # 単語数
	HIST_H_SIZE = hist_num # 文書数
	TERM_PER_SIZE = term_per_doc # ドキュメントごとの単語数
	MODE = mode # 不正するかしないか

	beta_w = [0.1 for i in range(HIST_W_SIZE)] # ディレクレ分布のパラメータ(グラフィカルモデル右端)
	beta_f = [0.1 for i in range(HIST_W_SIZE)] # ディレクレ分布のパラメータ(グラフィカルモデル右端)
	alpha = [0.1 for i in range(TOPIC_N)] # #ディレクレ分布のパラメータ(グラフィカルモデル左端)


	#print("alpha->",alpha)
	#print("beta->",beta)
	#FILE_NAME = sys.argv[1] # 保存先のファイル名
	FILE_NAME = "synthetic_data" # 保存先のファイル名

	hist_w = np.zeros( (HIST_H_SIZE, TERM_PER_SIZE) ) # ヒストグラム格納用の変数
	hist_f = np.zeros( (HIST_H_SIZE, TERM_PER_SIZE) ) # ヒストグラム格納用の変数
	label_z = np.zeros(HIST_H_SIZE) # 単語の潜在変数を元に文書ラベルを決定する変数
	z_max = -1145141919810 # z_countと比較するための変数
	tpd_t = int(TERM_PER_SIZE / TOPIC_N) # 不正操作用の変数．単語生成確率を操作
	prob = float(1 / tpd_t) # 不正操作用の変数．
	#print(document_label[0])
	#print(hist)
	#print(hist[0])

	# 各トピックの単語にわたる多項分布を生成
	phi_w = []
	phi_f = []
	topic_w = []
	topic_f = []
	for i in range(TOPIC_N):
		if MODE == True:
			topic_w = np.zeros(TERM_PER_SIZE)
			topic_w[i*tpd_t : tpd_t*(i+1)] = prob
			print(topic_w[::-1])
			phi_w.append(topic_w)
			phi_f.append(topic_w[::-1])
		else:
			topic_w = np.random.mtrand.dirichlet(beta_w, size = 1)
			topic_f = np.random.mtrand.dirichlet(beta_f, size = 1)
			phi_w.append(topic_w)
			phi_f.append(topic_f)
	print(f"phi_w->{phi_w}")
	print(f"phi_f->{phi_f}")



	# 各ファイル変数
	output_w = open(FILE_NAME+'.w','w')
	output_f = open(FILE_NAME+'.f','w')
	z_f = open(FILE_NAME+'.z_feature','w')
	theta_f = open(FILE_NAME+'.theta','w')

	hist_i = 0 # ヒストグラムの縦の要素のインデックス
	remove_label = [] # 潜在ラベルの重複インデックスを格納
	# 各ドキュメントの単語を生成
	for i in range(HIST_H_SIZE):
		print("epochs->{}".format(i))
		w_buffer = {}
		f_buffer = {}
		z_buffer = {} # 真のzをトラッキングするための変数
		theta = np.zeros((1,TOPIC_N), dtype = float)

		# θのサンプリング(トピック割当て確率を示す)
		if (MODE == True):
			theta[0][i%TOPIC_N] = 1.0
			# トピック比率の不正操作
		else:
			theta = np.random.mtrand.dirichlet(alpha,size = 1)
			# 一様分布

		for j in range(TERM_PER_SIZE):
			# zのサンプリング（生成されるトピック）
			z = np.random.multinomial(1,theta[0],size = 1)
			#print(f"theta -> {theta}")
			#print(f"z->{z[0]}")
			z_assignment = 0
			for k in range(TOPIC_N):
				if z[0][k] == 1:
					break
				z_assignment += 1
			if not z_assignment in z_buffer:
				z_buffer[z_assignment] = 0
			z_buffer[z_assignment] = z_buffer[z_assignment] + 1
			# トピックzからサンプリングされる観測w
			#print(f"z_assignment-> {z_assignment}")
			#print(f"phi_w-> {phi_w[z_assignment][0]}")
			#print(f"phi_w[z]-> {phi_w[z_assignment][0]}")
			if MODE == True:
				# 不正操作モード
				w = np.random.multinomial(1,phi_w[z_assignment],size = 1)
				f = np.random.multinomial(1,phi_f[z_assignment],size = 1)
			else:
				# 一様乱数モード
				w = np.random.multinomial(1,phi_w[z_assignment][0],size = 1)
				f = np.random.multinomial(1,phi_f[z_assignment][0],size = 1)
			#print(f"phi_w[z_assignment] -> {phi_w[z_assignment]}")
			#print(f"w -> {w[0]}")
			w_assignment = 0
			f_assignment = 0
			for k in range(HIST_W_SIZE):
				if w[0][k] == 1:
					break
				w_assignment += 1
			if not w_assignment in w_buffer:
				w_buffer[w_assignment] = 0
			w_buffer[w_assignment] = w_buffer[w_assignment] + 1
			for k in range(HIST_W_SIZE):
				if f[0][k] == 1:
					break
				f_assignment += 1
			if not f_assignment in f_buffer:
				f_buffer[f_assignment] = 0
			f_buffer[f_assignment] = f_buffer[f_assignment] + 1

		#print("------------------------------------------")

		"""
		ここまで人口データ作成
		"""
		# ここから各情報をファイルに保存
		output_w.write(str(i)+'\t'+str(TERM_PER_SIZE)+'\t')
		output_f.write(str(i)+'\t'+str(TERM_PER_SIZE)+'\t')
		for w_id, w_count in w_buffer.items():
			output_w.write(str(w_id)+':' + str(w_count) + ' ')
			hist_w[hist_i,w_id] = w_count
		output_w.write('\n')
		for f_id, f_count in f_buffer.items():
			output_f.write(str(f_id)+':' + str(f_count) + ' ')
			hist_f[hist_i,f_id] = f_count
		output_f.write('\n')
		z_f.write(str(i)+'\t'+str(TERM_PER_SIZE)+'\t')
		for z_id, z_count in z_buffer.items():
			if (z_max == z_count):
				remove_label.append(hist_i)
			if (z_max < z_count): # z_countが最大の時のz_idを文書ラベルとして採用する
				z_max = z_count
			label_z[hist_i] = z_id
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
	output_w.close()
	output_f.close()
	
	if test == False:
		np.savetxt( "k"+str(topic_n)+"tr_w.txt", hist_w, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"tr_f.txt", hist_f, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"tr_z.txt", label_z, fmt=str("%d") )
	else:
		np.savetxt( "k"+str(topic_n)+"te_w.txt", hist_w, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"te_f.txt", hist_f, fmt=str("%d") )
		np.savetxt( "k"+str(topic_n)+"te_z.txt", label_z, fmt=str("%d") )
	print("トピックが重複している文書",remove_label)
	print("不正操作モード選択->{}".format(MODE))
	print("テストデータ生成->{}".format(test))



if __name__ == "__main__":
	main()
