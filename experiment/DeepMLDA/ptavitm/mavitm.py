from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Optional, Tuple


def prior(topics: int, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prior for the model.

    :param topics: number of topics
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    a = torch.Tensor(1, topics).float().fill_(alpha) # 1 x 50 全て1.0
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / topics) * a.reciprocal()).t() + (1.0 / topics ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t()


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

def joint_encoder(input_x1_x2: int, # 入力次元数:x1の次元数+x2の次元数
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(input_x1_x2, hidden1_dimension)),
        ('act1', nn.Softplus()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Softplus()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))

def encoder(input_x1: int,
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(input_x1, hidden1_dimension)),
        ('act1', nn.Softplus()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Softplus()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))


def decoder(input_x1: int,
            topics: int,
            decoder_noise: float,
            eps: float,
            momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(topics, input_x1, bias=False)),
        ('batchnorm', nn.BatchNorm1d(input_x1, affine=True, eps=eps, momentum=momentum)),
        ('act', nn.Softmax(dim=1)),
        ('dropout', nn.Dropout(decoder_noise))
    ]))


def hidden(hidden2_dimension: int,
           topics: int,
           eps: float,
           momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(hidden2_dimension, topics)),
        ('batchnorm', nn.BatchNorm1d(topics, affine=True, eps=eps, momentum=momentum))
    ]))


class MAVITM(nn.Module):
    def __init__(self,
            topics: int,
            joint_input: int,
            input_x1: int,
            input_x2: int,
            hidden1_dimension: int,
            hidden2_dimension: int,
            decoder_noise: float = 0.2,
            encoder_noise: float = 0.2,
            batchnorm_eps: float = 0.001,
            batchnorm_momentum: float = 0.001,
            train_word_embeddings: bool = True,
            word_embeddings: Optional[Mapping[int, torch.Tensor]] = None) -> None:
        super(MAVITM, self).__init__()
        self.topics = topics
        """Define Inference Network and Generator """
        """注意：上の関数で作成"""
        # Inference net q(z|x1,x2)
        self.inference = joint_encoder(joint_input, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        # Inference net q(z|x1)
        self.inference_x1 = encoder(input_x1, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x1_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x1_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x1_generator = decoder(
            input_x1, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # Inference net q(z|x2)
        self.inference_x2 = encoder(input_x2, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x2_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x2_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x2_generator = decoder(
            input_x2, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # 事前分布のパラメータを定義, x1とx2の事前分布
        self.x1_prior_mean, self.x1_prior_var = map(nn.Parameter, prior(topics, 0.9))
        self.x1_prior_logvar = nn.Parameter(self.x1_prior_var.log())
        self.x1_prior_mean.requires_grad = False
        self.x1_prior_var.requires_grad = False
        self.x1_prior_logvar.requires_grad = False
        self.x2_prior_mean, self.x2_prior_var = map(nn.Parameter, prior(topics, 0.9))
        self.x2_prior_logvar = nn.Parameter(self.x2_prior_var.log())
        self.x2_prior_mean.requires_grad = False
        self.x2_prior_var.requires_grad = False
        self.x2_prior_logvar.requires_grad = False
        # do not learn the batchnorm weight, setting it to 1 as in https://git.io/fhtsY
        for component in [self.mean, self.logvar, self.x1_mean, self.x1_logvar, self.x2_mean, self.x2_logvar,  self.x1_generator, self.x2_generator]:
            component.batchnorm.weight.requires_grad = False
            component.batchnorm.weight.fill_(1.0)
        # エンコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.inference.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x1.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x2.linear1.weight, gain=1)
        if word_embeddings is not None:
            copy_embeddings_(self.inference.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x1.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x2.linear1.weight, word_embeddings)
        if not train_word_embeddings:
            self.inference.linear1.weight.requires_grad = False
            self.inference.linear1.bias.requires_grad = False
            self.inference.linear1.bias.fill_(0.0)
            self.inference_x1.linear1.weight.requires_grad = False
            self.inference_x1.linear1.bias.requires_grad = False
            self.inference_x1.linear1.bias.fill_(0.0)
            self.inference_x2.linear1.weight.requires_grad = False
            self.inference_x2.linear1.bias.requires_grad = False
            self.inference_x2.linear1.bias.fill_(0.0)
        # デコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.x1_generator.linear.weight, gain=1)
        nn.init.xavier_uniform_(self.x2_generator.linear.weight, gain=1)

    def joint_encode(self, x1_batch, x2_batch):
        """
        同時分布を求める推論ネットワーク用
        """
        input_data = torch.cat((x1_batch, x2_batch), 1)
        encoded = self.inference(input_data)
        return encoded, self.mean(encoded), self.logvar(encoded)

    def inferenceX1(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x1の推論ネットワーク
        """
        encoded = self.inference_x1(batch)
        return encoded, self.x1_mean(encoded), self.x1_logvar(encoded)


    def generatorX1(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        """
        x1の生成ネットワーク
        """
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        z = F.softmax(z,dim=1)
        return self.x1_generator(z)

    def inferenceX2(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x1とx2用の推論ネットワーク
        """
        encoded = self.inference_x2(batch)
        return encoded, self.x2_mean(encoded), self.x2_logvar(encoded)

    def generatorX2(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        """
        x2用の生成ネットワーク
        """
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        z = F.softmax(z,dim=1)
        return self.x2_generator(z)

    def sample_z(self,mean,logvar): # 独自で定義,潜在変数の可視化に必要
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        return mean + logvar.exp().sqrt() * eps

    def forward(self, x1_batch, x2_batch):
        _, mean, logvar = self.joint_encode(x1_batch,x2_batch) # 同時分布の平均と分散
        _, x1_mean, x1_logvar = self.inferenceX1(x1_batch) # x1モダリティの平均と分散
        _, x2_mean, x2_logvar = self.inferenceX2(x2_batch) # x2モダリティの平均と分散
        x1_recon = self.generatorX1(mean, logvar) # 同時分布から生成したx1モダリティの復元誤差
        x2_recon = self.generatorX2(mean, logvar) # 同時分布から生成したx2モダリティの復元誤差
        z_hoge = self.sample_z(mean, logvar)
        return mean, logvar, x1_mean, x1_logvar, x2_mean, x2_logvar, x1_recon, x2_recon, z_hoge

    def loss(self,
             input_tensor: torch.Tensor,
             reconstructed_tensor: torch.Tensor,
             posterior_mean: torch.Tensor,
             posterior_logvar: torch.Tensor,
             ) -> torch.Tensor:
        """
        Variational objective, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        https://arxiv.org/pdf/1703.01488.pdf; modified from https://github.com/hyqneuron/pytorch-avitm.

        :param input_tensor: input batch to the network, shape [batch size, features]
        :param reconstructed_tensor: reconstructed batch, shape [batch size, features]
        :param posterior_mean: posterior mean
        :param posterior_logvar: posterior log variance
        :return: unaveraged loss tensor

        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
        """
        # TODO check this again against original paper and TF implementation by adding tests
        # https://github.com/akashgit/autoencoding_vi_for_topic_models/blob/master/models/prodlda.py
        # reconstruction loss
        # this is the second line in Eq. 7

        rl = -(input_tensor * (reconstructed_tensor + 1e-10).log()).sum(1)
        # KL divergence
        # this is the first line in Eq. 7
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_logvar)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        var_division = posterior_logvar.exp() / prior_var # Σ_0 / Σ_1

        #print("posterior_mean->{}".format(posterior_mean))
        #print("prior_mean->{}".format(prior_mean))
        #print("posterior_var->{}".format(posterior_var))
        #print("prior_logvar->{}".format(prior_logvar))
        diff = posterior_mean - prior_mean # μ_１ - μ_0
        diff_term = diff * diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - posterior_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)

        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        """
        KL = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
           = 0.5 * {Σ_0 / Σ_1 + (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1 + log(|Σ_1|/|Σ_2|) -k}
        """
        loss = (rl + kld)
        return loss
