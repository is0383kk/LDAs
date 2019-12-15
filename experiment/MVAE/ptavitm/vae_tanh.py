from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Optional, Tuple

def copy_embeddings_(tensor: torch.Tensor, lookup: Mapping[int, torch.Tensor]) -> None:
    """
    Helper function for mutating the weight of an initial linear embedding module using
    precomputed word vectors.

    :param tensor: weight Tensor to mutate of shape [embedding_dimension, features]
    :param lookup: given an index return the corresponding Tensor
    :return: None
    """
    for index in range(tensor.shape[1]):
        current_embedding = lookup.get(index)
        if current_embedding is not None:
            tensor[:, index].copy_(current_embedding)


def encoder(in_dimension: int,
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(in_dimension, hidden1_dimension)),
        ('act1', nn.Tanh()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Tanh()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))


def decoder(in_dimension: int,
            topics: int,
            decoder_noise: float,
            eps: float,
            momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(topics, in_dimension, bias=False)),
        ('batchnorm', nn.BatchNorm1d(in_dimension, affine=True, eps=eps, momentum=momentum)),
        ('act', nn.Softmax()),
        ('dropout', nn.Dropout(decoder_noise)),
    ]))


def hidden(hidden2_dimension: int,
           topics: int,
           eps: float,
           momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(hidden2_dimension, topics)),
        ('batchnorm', nn.BatchNorm1d(topics, affine=True, eps=eps, momentum=momentum))
    ]))


class ProdLDA(nn.Module):
    def __init__(self,
                 in_dimension: int,
                 hidden1_dimension: int,
                 hidden2_dimension: int,
                 topics: int,
                 decoder_noise: float = 0.2,
                 encoder_noise: float = 0.2,
                 batchnorm_eps: float = 0.001,
                 batchnorm_momentum: float = 0.001,
                 train_word_embeddings: bool = True,
                 word_embeddings: Optional[Mapping[int, torch.Tensor]] = None) -> None:
        super(ProdLDA, self).__init__()
        self.topics = topics
        self.encoder = encoder(in_dimension, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.decoder = decoder(
            in_dimension, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # do not learn the batchnorm weight, setting it to 1 as in https://git.io/fhtsY
        for component in [self.mean, self.logvar, self.decoder]:
            component.batchnorm.weight.requires_grad = False
            component.batchnorm.weight.fill_(1.0)
        # エンコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.encoder.linear1.weight, gain=1)
        if word_embeddings is not None:
            copy_embeddings_(self.encoder.linear1.weight, word_embeddings)
        if not train_word_embeddings:
            self.encoder.linear1.weight.requires_grad = False
            self.encoder.linear1.bias.requires_grad = False
            self.encoder.linear1.bias.fill_(0.0)
        # デコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.decoder.linear.weight, gain=1)

    def encode(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        平均と分散を算出
        """
        encoded = self.encoder(batch)
        return encoded, self.mean(encoded), self.logvar(encoded)

    def decode(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        """
        リパラメトリゼーショントリック
        """
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        return self.decoder(z)

    def sample_z(self,mean,logvar): # 独自で定義,潜在変数の可視化に必要
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        return z

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, mean, logvar = self.encode(batch)
        recon = self.decode(mean, logvar)
        z_hoge = self.sample_z(mean, logvar)
        return recon, mean, logvar, z_hoge

    def loss(self,
             input_tensor: torch.Tensor,
             reconstructed_tensor: torch.Tensor,
             posterior_mean: torch.Tensor,
             posterior_logvar: torch.Tensor,
             topic: int,
             ) -> torch.Tensor:
        prior_mean = torch.zeros((topic))
        prior_var = torch.ones((topic))
        prior_logvar = prior_var.log()
        rl = -(input_tensor * (reconstructed_tensor + 1e-10).log()).sum(1)
        #print(f"input_tensor=>{input_tensor}")
        #print(f"reconstructed_tensor=>{reconstructed_tensor}")
        #print(f"recon_err=>{rl}")
        # KL divergence
        # this is the first line in Eq. 7
        #print(f"prior_mean->{prior_mean}")
        #print(f"prior_mean->{prior_var}")
        #print(f"prior_logvar->{prior_logvar}")
        var_division = posterior_logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = posterior_mean - prior_mean # μ_１ - μ_0
        diff_term = diff * diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - posterior_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        print("kld",kld)
        """
        KL = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
           = 0.5 * {Σ_0 / Σ_1 + (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1 + log(|Σ_1|/|Σ_2|) -k}
        """
        loss = (rl + kld)
        return loss
