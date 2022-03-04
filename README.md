# Implementation of LDAs
This repository is an implementation of LDA.  
Implementation contents are as follows　　
1. **Latent Dirichlet Allocation as a probabilistic generative model.**
2. **Multimodal Dirichlet Allocation as a probabilistic generative model.**
3. **Deep-LDA (LDA based on VAE) as a deep generative model.**
4. **Deep-MLDA (LDA based on JMVAE) as a deep generative model.**

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
