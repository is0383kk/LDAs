from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time


def train(dataset: torch.utils.data.Dataset,
          num_topics: int,
          autoencoder: torch.nn.Module,
          epochs: int,
          batch_size: int,
          optimizer: torch.optim.Optimizer,
          scheduler: Any = None,
          validation: Optional[torch.utils.data.Dataset] = None,
          corruption: Optional[float] = None,
          cuda: bool = False,
          sampler: Optional[torch.utils.data.sampler.Sampler] = None,
          silent: bool = False,
          update_freq: Optional[int] = 1,
          update_callback: Optional[Callable[[float, float], None]] = None,
          epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
          num_workers: int = 0,
          ) -> None:
    """
    Function to train an autoencoder using the provided dataset.

    :param dataset: training Dataset
    :param autoencoder: autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
    :param update_callback: optional function of loss and validation loss to update
    :param epoch_callback: optional function of epoch and model
    :param num_workers: optional number of workers for loader
    :return: None
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers
    )
    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
            num_workers=num_workers
        )
    else:
        validation_loader = None
    autoencoder.train()
    t1 = time.time()
    perplexity_value = -1
    loss_value = 0
    plt_epoch_list = np.arange(epochs)
    plt_loss_list = []
    plt_perplexity = []
    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit='batch',
            postfix={
                'epo': epoch,
                'lss': '%.6f' % 0.0,
                'ppx': '%.6f' % -1,
            },
            disable=silent,
        )
        losses = []
        for index, batch in enumerate(data_iterator):
            x1_batch = batch[0]
            x2_batch = batch[1]
            if cuda:
                x1_batch = x1_batch.cuda(non_blocking=True)
                x2_batch = x2_batch.cuda(non_blocking=True)
            # run the batch through the autoencoder and obtain the output
            if corruption is not None:
                mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge = autoencoder(F.dropout(x1_batch, x2_batch, corruption))
            else:
                mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge = autoencoder(x1_batch, x2_batch)
            # calculate the loss and backprop
            #jmvae_zero_loss = autoencoder.jmvae_zero_loss(x1_batch, x2_batch, mean, logvar, x1_recon, x1_mean, x1_logvar, x2_recon, x2_mean, x2_logvar).mean()
            #kl_x1_x2 = autoencoder.kl_x1_x2(mean, logvar, x1_mean, x1_logvar, x2_mean, x2_logvar).mean()
            #loss = autoencoder.jmvae_kl_loss(jmvae_zero_loss, kl_x1_x2, 1.0)
            #loss = jmvae_zero_loss
            loss = autoencoder.telbo(x1_batch, x2_batch, mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon,x2_mean ,x2_logvar, 1.0, 1.0, 1.0).mean()
            loss_value = float(loss.mean().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            # log losses
            losses.append(loss_value)
            data_iterator.set_postfix(
                epo=epoch,
                lss='%.6f' % loss_value,
                ppx='%.6f' % perplexity_value,
            )
        if update_freq is not None and epoch % update_freq == 0:
            average_loss = (sum(losses) / len(losses)) if len(losses) > 0 else -1
            #print("sum(losses)->",sum(losses))
            #print("len(losses)->",len(losses))
            if validation_loader is not None:
                autoencoder.eval()
                perplexity_value = perplexity(validation_loader, autoencoder, cuda, silent)
                data_iterator.set_postfix(
                    epo=epoch,
                    lss='%.6f' % average_loss,
                    ppx='%.6f' % perplexity_value,
                )
                autoencoder.train()
            else:
                perplexity_value = -1
                data_iterator.set_postfix(
                    epo=epoch,
                    lss='%.6f' % average_loss,
                    ppx='%.6f' % -1,
                )
            if update_callback is not None:
                update_callback(autoencoder, epoch, optimizer.param_groups[0]['lr'], average_loss, perplexity_value)
        if epoch_callback is not None:
            autoencoder.eval()
            epoch_callback(epoch, autoencoder)
            autoencoder.train()
        t2 = time.time()
        # 経過時間を表示
        elapsed_time = t2-t1
        print("実行時間",elapsed_time)
        #lossの可視化
        plt_loss_list.append(average_loss)
        plt_perplexity.append(perplexity_value)

    np.save('./runs/loss_list.npy', np.array(plt_loss_list))
    np.save('./runs/perplexity.npy', np.array(plt_perplexity))
    plt_loss_list = np.load('./runs/loss_list.npy')
    plt_perplexity = np.load('./runs/perplexity.npy')
    plt.figure(figsize=(13,9))
    plt.tick_params(labelsize=18)
    plt.title('Amortized MLDA(Topic='+ str(num_topics) +'):Log likelihood',fontsize=24)
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('Log likelihood',fontsize=24)
    plt.plot(plt_epoch_list,plt_loss_list)

    plt.savefig('liks.png')
    torch.save(autoencoder.state_dict(), 'deepmlda.pth')

def perplexity(loader: torch.utils.data.DataLoader, model: torch.nn.Module, cuda: bool = False, silent: bool = False):
    model.eval()
    data_iterator = tqdm(loader, leave=False, unit='batch', disable=silent)
    losses = []
    x1_counts = []
    x2_counts = []
    for index, batch in enumerate(data_iterator):
        x1_batch = batch[0]
        x2_batch = batch[1]
        if cuda:
            x1_batch = x1_batch.cuda(non_blocking=True)
            x2_batch = x2_batch.cuda(non_blocking=True)
        mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon, x2_mean, x2_logvar, z_hoge = model(x1_batch, x2_batch)
        #print(f'jmvae_zero_loss -> {jmvae_zero_loss}')
        #print(f'kl_x1_x2 -> {kl_x1_x2}')
        #losses.append(model.jmvae_zero_loss(x1_batch, x2_batch, mean, logvar, x1_recon, x1_mean, x1_logvar, x2_recon, x2_mean, x2_logvar).detach().cpu())
        losses.append(model.telbo(x1_batch, x2_batch, mean, logvar, jmvae_x1_recon, x1_recon, x1_mean, x1_logvar, jmvae_x2_recon, x2_recon,x2_mean ,x2_logvar, 1.0, 1.0, 1.0).detach().cpu())
        x1_counts.append(x1_batch.sum(1).detach().cpu())
        x2_counts.append(x2_batch.sum(1).detach().cpu())
        x1_pxp = float((torch.cat(losses) / torch.cat(x1_counts)).mean().exp().item())
        x2_pxp = float((torch.cat(losses) / torch.cat(x2_counts)).mean().exp().item())
        #print("torch.cat(counts)",torch.cat(counts))
        #print("len(counts)",len(counts))
        #print("torch.cat(losses)",torch.cat(losses))
        #print("losses",len(losses))
        #print("losses",losses)
        #print("(torch.cat(losses) / torch.cat(counts)).mean()",(torch.cat(losses) / torch.cat(counts)).mean())
        #print("(torch.cat(losses) / torch.cat(counts)).mean()",(torch.cat(losses) / torch.cat(counts)).mean().exp().item())
    return x1_pxp

def predict(dataset: torch.utils.data.Dataset,
            model: torch.nn.Module,
            batch_size: int,
            cuda: bool = False,
            silent: bool = False,
            encode: bool = True,
            num_workers: int = 0) -> torch.Tensor:
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.

    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :param num_workers: optional number of workers for loader
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        num_workers=num_workers
    )
    data_iterator = tqdm(
        dataloader,
        leave=False,
        unit='batch',
        disable=silent,
    )
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for index, batch in enumerate(data_iterator):
        batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        if encode:
            output = model.encode(batch)
            features.append(output[1].detach().cpu().exp())  # move to the CPU to prevent out of memory on the GPU
        else:
            output = model.forward(batch)
            features.append(output.detach().cpu())  # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)
