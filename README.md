# Implementation of LDAs
Ongoing
## Latent Dirichlet Allocation
[Original paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  

What this repo contains:  

- `/LDA/lda_gbs.py`:LDA by using Collapsed Gibbs sampler with Python
- `/LDA/bow/bow.py`: It can generate **BoW file** which is used by `/LDA/lda_gbs.py` from `/LDA/bow/text.txt`
- `/LDA/bow/text.txt`: You can write  sentences. **Sentences must be separated by spaces for each word.**
    - For example
    ```
    内藤 は 彼女 が できない
    内藤 は 理想 が 高すぎる
    あの 店 の ラーメン は おいしい
    おいしい 店 の ごはん を 食べる
    室 は ずっと 踊って いる
    室 は 顔芸 を して いる
    ```
- Usage
    1. Write sentences in `/LDA/bow/text.txt`
    2. `python3 /LDA/bow/bow.py`
    3. `python3 /LDA/lda_gbs.py`

- Requirement

Python3.X and the following modules are required

```python
import codecs
import numpy as np
from scipy.sparse import load_npz
import random
import math
import matplotlib.pyplot as plt
```

- [Implementation source of LDA](https://github.com/naka-tomo/LDA-PY)

# Multimodal Latent Dirichlet Allocation

What this repo contains:  

- `/MLDA/mlda.py`:MLDA by using Collapsed Gibbs sampler with Python
- `/MLDA/bow/bow.py`: It can generate **BoW file** which is used by `/MLDA/mlda.py` from `/MLDA/bow/text.txt`
- `/MLDA/bow/text.txt`: You can write  sentences. **Sentences must be separated by spaces for each word.**
    - For example
    ```
    内藤 は 彼女 が できない
    内藤 は 理想 が 高すぎる
    あの 店 の ラーメン は おいしい
    おいしい 店 の ごはん を 食べる
    室 は ずっと 踊って いる
    室 は 顔芸 を して いる
    ```

- `/MLDA/bof/bof.py`: It can generate **BoF file** which is used by `/MLDA/mlda.py` from `/MLDA/bof/images/*.png`

- Usage
    1. Write sentences in `/MLDA/bow/text.txt`
    2. Create BoW:`python3 /MLDA/bow/bow.py`
    3. Create BoF:`python3 /MLDA/bof/bof.py`
    4. `python3 /MLDA/mlda.py`

- Requirement

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
- [Implementation source of MLDA](https://github.com/naka-tomo/MLDA-PY)

# Autoencoded variational infarence for topic model

PyTorch implementation of the **Autoencoding Variational Inference For Topic Models (AVITM)** algorithm.
[Original paper](https://arxiv.org/abs/1703.01488)

- [Implementation source of AVITM by using PyTorch](https://github.com/vlukiyanov/pt-avitm)

- Requirement


- Other implementations of AVITM
    - [Original TensorFlow](https://github.com/akashgit/autoencoding_vi_for_topic_models)

    - [PyTorch(Old version)](https://github.com/hyqneuron/pytorch-avitm)
