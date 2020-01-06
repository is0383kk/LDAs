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

def joint_encoder(joint_input: int, # 入力次元数:x1の次元数+x2の次元数
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(joint_input, hidden1_dimension)),
        ('act1', nn.Tanh()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Tanh()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))

def encoder(input_x: int,
            hidden1_dimension: int,
            hidden2_dimension: int,
            encoder_noise: float = 0.2) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(input_x, hidden1_dimension)),
        ('act1', nn.Tanh()),
        ('linear2', nn.Linear(hidden1_dimension, hidden2_dimension)),
        ('act2', nn.Tanh()),
        ('dropout', nn.Dropout(encoder_noise))
    ]))

def decoder(input_x: int,
            topics: int,
            decoder_noise: float,
            eps: float,
            momentum: float) -> torch.nn.Module:
    return nn.Sequential(OrderedDict([
        ('linear', nn.Linear(topics, input_x, bias=False)),
        ('batchnorm', nn.BatchNorm1d(input_x, affine=True, eps=eps, momentum=momentum)),
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
        self.inference = joint_encoder(joint_input, 150, 150, encoder_noise)
        self.mean = hidden(150, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.logvar = hidden(150, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        # Inference net q(z|x1) & Generator p(x1|z)
        self.inference_x1 = encoder(input_x1, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x1_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x1_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x1_generator = decoder(
            input_x1, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # Inference net q(z|x2) & Generator p(x2|z)
        self.inference_x2 = encoder(input_x2, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x2_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x2_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x2_generator = decoder(
            input_x2, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # 事前分布のパラメータを定義
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(topics, 0.995))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
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
        encoded = self.inference(torch.cat([x1_batch, x2_batch], 1))
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
        x2用の推論ネットワーク
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
        jmvae_x1_recon = self.generatorX1(mean, logvar) # 同時分布から生成したx1モダリティの復元誤差
        jmvae_x2_recon = self.generatorX2(mean, logvar) # 同時分布から生成したx2モダリティの復元誤差
        x1_recon = self.generatorX1(x1_mean, x1_logvar) # 同時分布から生成したx1モダリティの復元誤差
        x2_recon = self.generatorX2(x2_mean, x2_logvar) # 同時分布から生成したx2モダリティの復元誤差
        z_hoge = self.sample_z(mean, logvar)
        return mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge


    def telbo(self,
             x1_batch: torch.Tensor,
             x2_batch: torch.Tensor,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             jmvae_x1_recon: torch.Tensor,
             x1_recon: torch.Tensor,
             x1_mean: torch.Tensor,
             x1_logvar: torch.Tensor,
             jmvae_x2_recon: torch.Tensor,
             x2_recon: torch.Tensor,
             x2_mean: torch.Tensor,
             x2_logvar: torch.Tensor,
             lambda_x1x2: float,
             lambda_x1: float,
             lambda_x2: float,
             ) -> torch.Tensor:
        jmvae_x1_rl = -(x1_batch * (jmvae_x1_recon + 1e-10).log()).sum(1)
        jmvae_x2_rl = -(x2_batch * (jmvae_x2_recon + 1e-10).log()).sum(1)
        x1_rl = -(x1_batch * (x1_recon + 1e-10).log()).sum(1)
        x2_rl = -(x2_batch * (x2_recon + 1e-10).log()).sum(1)
        ##############################################################################################
        # 同時分布と事前分布とのKL計算
        prior_mean = self.prior_mean.expand_as(mean)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mean - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        # x1のKL
        x1_prior_mean = self.prior_mean.expand_as(x1_mean)
        x1_prior_var = self.prior_var.expand_as(x1_logvar)
        x1_prior_logvar = self.prior_logvar.expand_as(x1_logvar)
        x1_var_division = x1_logvar.exp() / x1_prior_var # Σ_0 / Σ_1
        x1_diff = x1_mean - x1_prior_mean # μ_１ - μ_0
        x1_diff_term = x1_diff *x1_diff / x1_prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        x1_logvar_division = x1_prior_logvar - x1_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        x1_kld = 0.5 * ((x1_var_division + x1_diff_term + x1_logvar_division).sum(1) - self.topics)
        # x2のKL
        x2_prior_mean = self.prior_mean.expand_as(x2_mean)
        x2_prior_var = self.prior_var.expand_as(x2_logvar)
        x2_prior_logvar = self.prior_logvar.expand_as(x2_logvar)
        x2_var_division = x2_logvar.exp() / x2_prior_var # Σ_0 / Σ_1
        x2_diff = x2_mean - x2_prior_mean # μ_１ - μ_0
        x2_diff_term = x2_diff *x2_diff / x2_prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        x2_logvar_division = x2_prior_logvar - x2_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        x2_kld = 0.5 * ((x2_var_division + x2_diff_term + x2_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # JMVAE_loss, TELBO_lossの第１項
        jmvae_zero_loss = kld + lambda_x1x2 * (jmvae_x1_rl + jmvae_x2_rl) # Equation (3) of paper of JMVAE
        x1_elbo = x1_kld + lambda_x1 * x1_rl
        x2_elbo = x2_kld + lambda_x2 * x2_rl
        return jmvae_zero_loss + x1_elbo + x2_elbo

    def jmvae_zero_loss(self,
             x1_batch: torch.Tensor,
             x2_batch: torch.Tensor,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             x1_recon: torch.Tensor,
             x1_mean: torch.Tensor,
             x1_logvar: torch.Tensor,
             x2_recon: torch.Tensor,
             x2_mean: torch.Tensor,
             x2_logvar: torch.Tensor,
             ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1611.01891.pdf
        This is MAVITM's loss function based on JMVAE(zero)
        JMVAEに基づいて目的関数を定義
        """
        x1_rl = -(x1_batch * (x1_recon + 1e-10).log()).sum(1)
        x2_rl = -(x2_batch * (x2_recon + 1e-10).log()).sum(1)
        ##############################################################################################
        # 同時分布と事前分布とのKL計算
        prior_mean = self.prior_mean.expand_as(mean)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mean - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # JMVAE_loss
        jmvae_zero_loss = (kld + x1_rl + x2_rl) # Equation (3) of paper
        return jmvae_zero_loss

    def kl_x1_x2(self,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             x1_mean: torch.Tensor,
             x1_logvar: torch.Tensor,
             x2_mean: torch.Tensor,
             x2_logvar: torch.Tensor,
             ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1611.01891.pdf
        This is MAVITM's loss function based on JMVAE(zero)
        JMVAEに基づいて目的関数を定義
        """
        ##############################################################################################
        """
        https://arxiv.org/pdf/1611.01891.pdf
        """
        # q(z|x1,x2) and q(z|x1)
        x1_var_division = logvar.exp() / x1_logvar.exp()
        x1_diff = mean - x1_mean
        x1_diff_term = x1_diff *x1_diff / x1_logvar.exp()
        x1_logvar_division = x1_logvar - logvar
        # KL q(z|x1,x2) and q(z|x1)
        x1_kld = 0.5 * ((x1_var_division + x1_diff_term + x1_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # q(z|x1,x2) and q(z|x2)
        x2_var_division = logvar.exp() / x2_logvar.exp()
        x2_diff = mean - x2_mean
        x2_diff_term = x2_diff *x2_diff / x2_logvar.exp()
        x2_logvar_division = x2_logvar - logvar
        # KL q(z|x1,x2) and q(z|x2)
        x2_kld = 0.5 * ((x2_var_division + x2_diff_term + x2_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # Equation (4) of paper
        #print(f"x1_kld -> {x1_kld}")
        #print(f"x2_kld -> {x2_kld}")
        #print(f"x1_kld + x2_kld -> {x1_kld + x2_kld}")
        return x1_kld + x2_kld

    def jmvae_kl_loss(self,
             jmvae_zero_loss: torch.Tensor,
             kl_x1_x2: torch.Tensor,
             alpha: float,
             ) -> torch.Tensor:
        loss = jmvae_zero_loss + (alpha * kl_x1_x2)
        return loss
