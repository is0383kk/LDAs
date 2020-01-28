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
            input_x3: int,
            input_x4: int,
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
        self.inference = joint_encoder(joint_input, 350, 350, encoder_noise)
        self.mean = hidden(350, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.logvar = hidden(350, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
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
        # Inference net q(z|x3) & Generator p(x3|z)
        self.inference_x3 = encoder(input_x3, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x3_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x3_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x3_generator = decoder(
            input_x3, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )
        # Inference net q(z|x4) & Generator p(x4|z)
        self.inference_x4 = encoder(input_x4, hidden1_dimension, hidden2_dimension, encoder_noise)
        self.x4_mean = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x4_logvar = hidden(hidden2_dimension, topics, eps=batchnorm_eps, momentum=batchnorm_momentum)
        self.x4_generator = decoder(
            input_x4, topics, decoder_noise=decoder_noise, eps=batchnorm_eps, momentum=batchnorm_momentum
        )

        # 事前分布のパラメータを定義
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(topics, 0.55))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
        # do not learn the batchnorm weight, setting it to 1 as in https://git.io/fhtsY
        for component in [self.mean, self.logvar,
                          self.x1_mean, self.x1_logvar, self.x1_generator,
                          self.x2_mean, self.x2_logvar, self.x2_generator,
                          self.x3_mean, self.x3_logvar, self.x3_generator,
                          self.x4_mean, self.x4_logvar, self.x4_generator
                          ]:
            component.batchnorm.weight.requires_grad = False
            component.batchnorm.weight.fill_(1.0)
        # エンコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.inference.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x1.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x2.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x3.linear1.weight, gain=1)
        nn.init.xavier_uniform_(self.inference_x4.linear1.weight, gain=1)
        if word_embeddings is not None:
            copy_embeddings_(self.inference.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x1.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x2.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x3.linear1.weight, word_embeddings)
            copy_embeddings_(self.inference_x4.linear1.weight, word_embeddings)
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
            self.inference_x3.linear1.weight.requires_grad = False
            self.inference_x3.linear1.bias.requires_grad = False
            self.inference_x3.linear1.bias.fill_(0.0)
            self.inference_x4.linear1.weight.requires_grad = False
            self.inference_x4.linear1.bias.requires_grad = False
            self.inference_x4.linear1.bias.fill_(0.0)
        # デコーダの重みを初期化（Xavierの初期値）
        nn.init.xavier_uniform_(self.x1_generator.linear.weight, gain=1)
        nn.init.xavier_uniform_(self.x2_generator.linear.weight, gain=1)
        nn.init.xavier_uniform_(self.x3_generator.linear.weight, gain=1)
        nn.init.xavier_uniform_(self.x4_generator.linear.weight, gain=1)

    def joint_encode(self, x1_batch, x2_batch, x3_batch, x4_batch):
        """
        同時分布を求める推論ネットワーク用
        """
        encoded = self.inference(torch.cat([x1_batch, x2_batch, x3_batch, x4_batch], 1))
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

    def inferenceX3(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x3の推論ネットワーク
        """
        encoded = self.inference_x3(batch)
        return encoded, self.x3_mean(encoded), self.x3_logvar(encoded)

    def generatorX3(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        """
        x3の生成ネットワーク
        """
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        z = F.softmax(z,dim=1)
        return self.x3_generator(z)

    def inferenceX4(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x4の推論ネットワーク
        """
        encoded = self.inference_x4(batch)
        return encoded, self.x4_mean(encoded), self.x4_logvar(encoded)

    def generatorX4(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Same as _generator_network method in the original TensorFlow implementation at https://git.io/fhUJu
        """
        x4の生成ネットワーク
        """
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        z = mean + logvar.exp().sqrt() * eps
        z = F.softmax(z,dim=1)
        return self.x4_generator(z)

    def sample_z(self,mean,logvar): # 独自で定義,潜在変数の可視化に必要
        eps = mean.new().resize_as_(mean).normal_(mean=0, std=1)
        return mean + logvar.exp().sqrt() * eps

    def forward(self, x1_batch, x2_batch, x3_batch, x4_batch):
        _, mean, logvar = self.joint_encode(x1_batch, x2_batch, x3_batch, x4_batch) # 同時分布の平均と分散
        _, x1_mean, x1_logvar = self.inferenceX1(x1_batch) # x1モダリティの平均と分散
        _, x2_mean, x2_logvar = self.inferenceX2(x2_batch) # x2モダリティの平均と分散
        _, x3_mean, x3_logvar = self.inferenceX3(x3_batch) # x3モダリティの平均と分散
        _, x4_mean, x4_logvar = self.inferenceX4(x4_batch) # x4モダリティの平均と分散
        jmvae_x1_recon = self.generatorX1(mean, logvar) # 同時分布から生成したx1モダリティの復元誤差
        jmvae_x2_recon = self.generatorX2(mean, logvar) # 同時分布から生成したx2モダリティの復元誤差
        jmvae_x3_recon = self.generatorX3(mean, logvar) # 同時分布から生成したx3モダリティの復元誤差
        jmvae_x4_recon = self.generatorX4(mean, logvar) # 同時分布から生成したx4モダリティの復元誤差
        x1_recon = self.generatorX1(x1_mean, x1_logvar) # 同時分布から生成したx1モダリティの復元誤差
        x2_recon = self.generatorX2(x2_mean, x2_logvar) # 同時分布から生成したx2モダリティの復元誤差
        x3_recon = self.generatorX3(x3_mean, x3_logvar) # 同時分布から生成したx3モダリティの復元誤差
        x4_recon = self.generatorX4(x4_mean, x4_logvar) # 同時分布から生成したx4モダリティの復元誤差
        z_hoge = self.sample_z(mean, logvar)
        return mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, jmvae_x3_recon, x3_recon, x3_mean, x3_logvar, jmvae_x4_recon, x4_recon, x4_mean, x4_logvar, z_hoge


    def telbo(self,
             x1_batch: torch.Tensor,
             x2_batch: torch.Tensor,
             x3_batch: torch.Tensor,
             x4_batch: torch.Tensor,
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
             jmvae_x3_recon: torch.Tensor,
             x3_recon: torch.Tensor,
             x3_mean: torch.Tensor,
             x3_logvar: torch.Tensor,
             jmvae_x4_recon: torch.Tensor,
             x4_recon: torch.Tensor,
             x4_mean: torch.Tensor,
             x4_logvar: torch.Tensor,
             lambda_x1234: float,
             lambda_x1: float,
             lambda_x2: float,
             lambda_x3: float,
             lambda_x4: float,
             ) -> torch.Tensor:
        jmvae_x1_rl = -(x1_batch * (jmvae_x1_recon + 1e-10).log()).sum(1)
        jmvae_x2_rl = -(x2_batch * (jmvae_x2_recon + 1e-10).log()).sum(1)
        jmvae_x3_rl = -(x3_batch * (jmvae_x3_recon + 1e-10).log()).sum(1)
        jmvae_x4_rl = -(x4_batch * (jmvae_x4_recon + 1e-10).log()).sum(1)
        x1_rl = -(x1_batch * (x1_recon + 1e-10).log()).sum(1)
        x2_rl = -(x2_batch * (x2_recon + 1e-10).log()).sum(1)
        x3_rl = -(x3_batch * (x3_recon + 1e-10).log()).sum(1)
        x4_rl = -(x4_batch * (x4_recon + 1e-10).log()).sum(1)
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
        x3_prior_mean = self.prior_mean.expand_as(x3_mean)
        x3_prior_var = self.prior_var.expand_as(x3_logvar)
        x3_prior_logvar = self.prior_logvar.expand_as(x1_logvar)
        x3_var_division = x3_logvar.exp() / x3_prior_var # Σ_0 / Σ_1
        x3_diff = x3_mean - x3_prior_mean # μ_１ - μ_0
        x3_diff_term = x3_diff *x3_diff / x3_prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        x3_logvar_division = x3_prior_logvar - x3_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        x3_kld = 0.5 * ((x3_var_division + x3_diff_term + x3_logvar_division).sum(1) - self.topics)
        #############################################################################################
        x4_prior_mean = self.prior_mean.expand_as(x4_mean)
        x4_prior_var = self.prior_var.expand_as(x4_logvar)
        x4_prior_logvar = self.prior_logvar.expand_as(x4_logvar)
        x4_var_division = x4_logvar.exp() / x4_prior_var # Σ_0 / Σ_1
        x4_diff = x4_mean - x4_prior_mean # μ_１ - μ_0
        x4_diff_term = x4_diff *x4_diff / x4_prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        x4_logvar_division = x4_prior_logvar - x4_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        x4_kld = 0.5 * ((x4_var_division + x4_diff_term + x4_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # JMVAE_loss, TELBO_lossの第１項
        jmvae_zero_loss = kld + lambda_x1234 * (jmvae_x1_rl + jmvae_x2_rl + jmvae_x3_rl + jmvae_x4_rl) # Equation (3) of paper of JMVAE
        x1_elbo = x1_kld + lambda_x1 * x1_rl
        x2_elbo = x2_kld + lambda_x2 * x2_rl
        x3_elbo = x3_kld + lambda_x3 * x3_rl
        x4_elbo = x4_kld + lambda_x4 * x4_rl
        return jmvae_zero_loss + x1_elbo + x2_elbo + x3_elbo + x4_elbo

    def jmvae_zero_loss(self,
             x1_batch: torch.Tensor,
             x2_batch: torch.Tensor,
             x3_batch: torch.Tensor,
             x4_batch: torch.Tensor,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             x1_recon: torch.Tensor,
             x2_recon: torch.Tensor,
             x3_recon: torch.Tensor,
             x4_recon: torch.Tensor,
             ) -> torch.Tensor:
        """
        https://arxiv.org/pdf/1611.01891.pdf
        This is MAVITM's loss function based on JMVAE(zero)
        JMVAEに基づいて目的関数を定義
        """
        x1_rl = -(x1_batch * (x1_recon + 1e-10).log()).sum(1)
        x2_rl = -(x2_batch * (x2_recon + 1e-10).log()).sum(1)
        x3_rl = -(x3_batch * (x3_recon + 1e-10).log()).sum(1)
        x4_rl = -(x4_batch * (x4_recon + 1e-10).log()).sum(1)
        ##############################################################################################
        # 同時分布と事前分布とのKL計算
        prior_mean = self.prior_mean.expand_as(mean) # 事前分布
        prior_var = self.prior_var.expand_as(logvar) #　事前分布
        prior_logvar = self.prior_logvar.expand_as(logvar) #事前分布
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mean - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        kld = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # JMVAE_loss
        jmvae_zero_loss = (kld + x1_rl + x2_rl + x3_rl + x4_rl) # Equation (3) of paper
        return jmvae_zero_loss

    def kl_x1234(self,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             x1_mean: torch.Tensor,
             x1_logvar: torch.Tensor,
             x2_mean: torch.Tensor,
             x2_logvar: torch.Tensor,
             x3_mean: torch.Tensor,
             x3_logvar: torch.Tensor,
             x4_mean: torch.Tensor,
             x4_logvar: torch.Tensor,
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
        # q(z|x1,x2,x3,x4) and q(z|x1)
        x1_var_division = logvar.exp() / x1_logvar.exp()
        x1_diff = mean - x1_mean
        x1_diff_term = x1_diff *x1_diff / x1_logvar.exp()
        x1_logvar_division = x1_logvar - logvar
        # KL q(z|x1,x2,x3,x4) and q(z|x1)
        x1_kld = 0.5 * ((x1_var_division + x1_diff_term + x1_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # q(z|x1,x2,x3,x4) and q(z|x2)
        x2_var_division = logvar.exp() / x2_logvar.exp()
        x2_diff = mean - x2_mean
        x2_diff_term = x2_diff *x2_diff / x2_logvar.exp()
        x2_logvar_division = x2_logvar - logvar
        # KL q(z|x1,x2,x3,x4) and q(z|x2)
        x2_kld = 0.5 * ((x2_var_division + x2_diff_term + x2_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # q(z|x1,x2,x3,x4) and q(z|x3)
        x3_var_division = logvar.exp() / x3_logvar.exp()
        x3_diff = mean - x3_mean
        x3_diff_term = x3_diff *x3_diff / x3_logvar.exp()
        x3_logvar_division = x3_logvar - logvar
        # KL q(z|x1,x2,x3,x4) and q(z|x3)
        x3_kld = 0.5 * ((x3_var_division + x3_diff_term + x3_logvar_division).sum(1) - self.topics)
        ##############################################################################################
        # q(z|x1,x2,x3,x4) and q(z|x3)
        x4_var_division = logvar.exp() / x4_logvar.exp()
        x4_diff = mean - x4_mean
        x4_diff_term = x4_diff *x4_diff / x4_logvar.exp()
        x4_logvar_division = x4_logvar - logvar
        # KL q(z|x1,x2,x3,x4) and q(z|x3)
        x4_kld = 0.5 * ((x4_var_division + x4_diff_term + x4_logvar_division).sum(1) - self.topics)
        ##############################################################################################

        return x1_kld + x2_kld + x3_kld + x4_kld

    def jmvae_kl_loss(self,
             jmvae_zero_loss: torch.Tensor,
             kl_x1234: torch.Tensor,
             alpha: float,
             ) -> torch.Tensor:
        loss = jmvae_zero_loss + (alpha * kl_x1234)
        return loss
