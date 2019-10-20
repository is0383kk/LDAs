# 今後の課題
1. inference networkの構造
NNのパラメータ等は把握しておく
2層だと分布形状が複雑になる場合，うまく分離できなくなる可能性がある
2. LDAのGibbs samplerと比較してどうか？
100文書以上で速くなっている
計算時間の比較を載せる
イテレーションの種類が違うことに注意
3. 学習過程の可視化,確率的モデルとして実装できているかどうかの確認として尤度の可視化を行う
パープレキシティでの評価
Gibbs sampling とVI（変分推論）で比較
トレーニングとテストを分けてやってみる
4. NNのネットワークの構造を3種類くらい用意してやってみる
浅いやつ、深いやつ
2層だとできないことを見る

# Implementation of LDAs
This repository is an implementation of LDA.  
Implementation contents are as follows　　
1. **Latent Dirichlet Allocation as a probabilistic generative model.**
2. **Multimodal Dirichlet Allocation as a probabilistic generative model.**
3. **Deep-LDA as a deep generative model.**
4. **Deep-MLDA as a deep generative model.**

# Latent Dirichlet Allocation
[Original paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  

What the `LDA` contains:  

- `/LDA/lda.py`:LDA by using Collapsed Gibbs sampler with Python.You need to decide the number of `topic`.
```python
def main():
    n = 100 # データの水増し用の変数
    topic = 3 # トピック数を指定
    data = np.loadtxt( root , dtype=np.int32)*n # 発生回数にnをかけて水増し可能
    #print(data)
    lda( data , topic )
```
**※Before running this script, you need to run  `/LDA/bow/bow.py` which creates a BoW file**
- `/LDA/bow/bow.py`: It can generate **BoW file** which is used by `/LDA/lda.py` from `/LDA/bow/text.txt`
- `/LDA/bow/text.txt`: You can write  sentences.   
**※Sentences must be separated by spaces for each word.**
    - *For example*
    ```
    内藤 は 彼女 が できない
    内藤 は 理想 が 高すぎる
    あの 店 の ラーメン は おいしい
    おいしい 店 の ごはん を 食べる
    室 は ずっと 踊って いる
    室 は 顔芸 を して いる
    ```
**Usage:**  
1. Write sentences in `/LDA/bow/text.txt`  
2. Create BoW:`python3 /LDA/bow/bow.py`  
3. `python3 /LDA/lda.py`  

**Requirement**  
Python3.X and the following modules are required  
```python
import codecs
import numpy as np
from scipy.sparse import load_npz
import random
import math
import matplotlib.pyplot as plt
```
**References:**
- [Implementation source of LDA](https://github.com/naka-tomo/LDA-PY)

# Multimodal Latent Dirichlet Allocation

**What the `MLDA` contains:**  

- `/MLDA/mlda.py`:MLDA by using Collapsed Gibbs sampler with Python.You need to decide the number of `topic`.
```python
def main():
    topic = 3
    data = []
    data.append( np.loadtxt( "./bof/histogram_v.txt" , dtype=np.int32) )
    data.append( np.loadtxt( "./bow/histogram_w.txt" , dtype=np.int32)*5 )
    mlda( data, topic, 100, "learn_result" )

    data[1] = None
    mlda( data, topic, 10, "recog_result" , "learn_result" )
```
**※Before running this script, you need to run `/MLDA/bow/bow.py` and `/MLDA/bof/bof.py` which create BoW and BoF file**
- `/MLDA/bow/bow.py`: It can generate **BoW file** which is used by `/MLDA/mlda.py` from `/MLDA/bow/text.txt`
- `/MLDA/bow/text.txt`: You can write  sentences.  
**※Sentences must be separated by spaces for each word.**
    - *For example*
    ```
    内藤 は 彼女 が できない
    内藤 は 理想 が 高すぎる
    あの 店 の ラーメン は おいしい
    おいしい 店 の ごはん を 食べる
    室 は ずっと 踊って いる
    室 は 顔芸 を して いる
    ```

- `/MLDA/bof/bof.py`: It can generate **BoF file** which is used by `/MLDA/mlda.py` from `/MLDA/bof/images/*.png`

**Usage:**  
1. Write sentences in `/MLDA/bow/text.txt`  
2. Create BoW:`python3 /MLDA/bow/bow.py`  
3. Create BoF:`python3 /MLDA/bof/bof.py`  
4. `python3 /MLDA/mlda.py`  

**Requirement:**  

Python3.X and the following modules are required

```python
import numpy as np
import random
import math
import pylab
import pickle
import os
import sys
import cv2
import glob
import codecs
```
**References:**
- [Implementation source of MLDA](https://github.com/naka-tomo/MLDA-PY)

# Autoencoded variational infarence for topic model

PyTorch implementation of the **Autoencoding Variational Inference For Topic Models (AVITM)** algorithm.
[Original paper](https://arxiv.org/abs/1703.01488)

**What the `DeepLDA` contains:**  

- `/DeepLDA/deeplda_few.py`:This is DeepLDA based on AVITM for `/few_input_data/`.
    - `/DeepLDA/few_input_data/text.txt`: You can write  sentences.  
     **※Sentences must be separated by spaces for each word.**
        - *For example*
        ```
        内藤 は 彼女 が できない
        内藤 は 理想 が 高すぎる
        あの 店 の ラーメン は おいしい
        おいしい 店 の ごはん を 食べる
        室 は ずっと 踊って いる
        室 は 顔芸 を して いる
        ```
    - `/DeepLDA/module/bow.py`:This code creates a BoW histogram, but uses it as a function within `deeplda_few.py`.

- `/DeepLDA/deeplda_huge.py`:This is DeepLDA based on AVITM for `/huge_input_data/`.
    - `/DeepLDA/huge_input_data/*_wine.csv`:This example uses the original data from the Kaggle dataset which can be found at [this page](https://www.kaggle.com/zynicide/wine-reviews).  
    - `/DeepLDA/prepare_huge_data.py`:This creates `test.txt.npz`,`train.txt.npz` and `vocab.pkl` to do with `deeplda_huge.py` from `DeepLDA/huge_input_data/*.csv`.
        - `vocab.pkl`:All words in `/DeepLDA/huge_input_data/*.csv` file are stored.
- `DeepLDA/ptavitm/`:VAE models and training functions used in `/DeepLDA/deeplda_few.py`,`/DeepLDA/deeplda_huge.py` are defined.  

**Requirement:**

Python3.X and the following modules are required.  
**PyTorch =>1.0**
```python
import click
import numpy as np
from torch.optim import Adam
import torch
from torch.utils.data import TensorDataset
from tensorboardX import SummaryWriter
import pickle
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus
from scipy.sparse import load_npz
from os.path import isfile, join
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz
import textacy
import os
import sys
import codecs
import cv2
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import torch.nn as nn
from typing import Mapping, Optional, Tuple
from torch.utils.data import Dataset
```

**Usage:**  
Ongoing

**References:**
- [Implementation source of AVITM by using PyTorch](https://github.com/vlukiyanov/pt-avitm)
- Other implementations of AVITM
    - [Original TensorFlow](https://github.com/akashgit/autoencoding_vi_for_topic_models)

    - [PyTorch(Old version)](https://github.com/hyqneuron/pytorch-avitm)

# Multimodal autoencoding variational infarence for topic model
Ongoing
