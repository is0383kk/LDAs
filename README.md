# Implementation of LDAs
Ongoing
## Latent Dirichlet Allocation
- `LDA/lda_gbs.py`:LDA by using Collapsed Gibbs sampler with Python
- `LDA/txtBoW_light/text.txt`: You can write  sentences. **Sentences must be separated by spaces for each word.**
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
    1. Write sentences in `LDA/txtBoW_light/text.txt`
    2. `python3 /LDA/txtBoW_light/bow.py`
    3. `python3 /LDA/lda_gbs.py`

- Requirement

Python3.X

```python
import codecs
import numpy
from scipy.sparse import load_npz
import random
import math
import matplotlib.pyplot as plt
```

- [Implementation source of LDA](https://github.com/naka-tomo/LDA-PY)

# Multimodal Latent Dirichlet Allocation
- [Implementation source of MLDA](https://github.com/naka-tomo/MLDA-PY)

# Autoencoded variational infarence for topic model

PyTorch implementation of the **Autoencoding Variational Inference For Topic Models (AVITM)** algorithm.
[Original paper](https://arxiv.org/abs/1703.01488)

- [Implementation source of AVITM by using PyTorch](https://github.com/vlukiyanov/pt-avitm)

- Requirement


- Other implementations of AVITM
    - [Original TensorFlow](https://github.com/akashgit/autoencoding_vi_for_topic_models)

    - [PyTorch(Old version)](https://github.com/hyqneuron/pytorch-avitm)
