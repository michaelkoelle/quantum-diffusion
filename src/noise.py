"""
Different methods of adding noise to data
"""

import torch
from einops import rearrange, repeat, reduce


def l1_norm(data):
    return torch.nn.functional.normalize(data, p=1, dim=-1)


def l2_norm(data):
    return torch.nn.functional.normalize(data, p=2, dim=-1)


def normalize_mean(target_data, inp):
    """
    Normalize the noisy data to have the same mean as the original data.
    shape of target_data (batch pixels).
    shape of inp (tau batch pixels) or ((batch tau) pixels).
    """
    if target_data.dim() == 1:
        target_data = target_data.unsqueeze(0)
    btp = False
    if inp.dim() == 2:
        btp = True
        batch_size = target_data.shape[0]
        inp = rearrange(inp, "(batch tau) pixels -> tau batch pixels", batch=batch_size)
    tau = inp.shape[0]
    inp_mean = reduce(inp, "tau batch pixels -> tau batch 1", "mean")
    orig_mean = reduce(target_data, "batch pixels -> batch 1", "mean")
    orig_mean = repeat(orig_mean, "batch 1 -> tau batch 1", tau=tau)
    moved = inp / inp_mean * orig_mean
    if btp:
        moved = rearrange(moved, "tau batch pixels -> (batch tau) pixels")
    return moved


def add_uniform_noise_iteratively(data, tau, decay_mod=1.0):
    """
    Add noise to data iteratively. 
    In the returned tensor, the first row is the original data, \
    the second row is the first row with additional noise, \
    the third row is the second row with additional noise, etc. 
    """
    if data.dim() == 1:
        data = data.unsqueeze(0)
    batch, pixels = data.shape
    noise_weighting = torch.linspace(0, 1, tau).to(data.device) ** decay_mod
    noise_weighting = l2_norm(noise_weighting)  # normalize
    noisy_data = torch.zeros(tau, batch, pixels).to(data.device)
    noisy_data[0] = data
    for it in range(1, tau):
        noise = torch.rand((batch, pixels), dtype=torch.double)
        new_data = (
            noisy_data[it - 1] * (1 - noise_weighting[it]) + noise * noise_weighting[it]
        )
        noisy_data[it] = new_data
    noisy_data = rearrange(noisy_data, "tau batch pixels -> (batch tau) pixels")
    return noisy_data


def add_uniform_noise_multiple(data, tau, decay_mod=2.0):
    """
    Add noise to data multiple times.
    The same noise is added in different weights to each image in the batch.
    """
    if data.dim() == 1:
        data = data.unsqueeze(0)
    batch, pixels = data.shape
    noise = torch.rand((batch, pixels), dtype=torch.double).to(data.device)
    data_expanded = repeat(data, "batch pixels -> tau batch pixels", tau=tau)
    noise_expanded = repeat(noise, "batch pixels -> tau batch pixels", tau=tau)
    noise_weighting = torch.linspace(0, 1, tau).to(data.device) ** decay_mod
    noise_weighting = noise_weighting / noise_weighting.max()  # normalize
    noise_weighting = repeat(noise_weighting, "tau -> tau batch 1", batch=batch)
    noisy_data = (
        data_expanded * (1 - noise_weighting) + noise_expanded * noise_weighting
    )
    noisy_data = rearrange(noisy_data, "tau batch pixels -> (batch tau) pixels")
    return noisy_data


def add_noise_normal_iteratively(data, tau, decay_mod=0.4):
    """
    Distorting the data iteratively by sampling from a normal distribution
    with the previous data as mean and a growing standard deviation.
    """
    if data.dim() == 1:
        data = data.unsqueeze(0)
    batch, pixels = data.shape
    noise_weighting = torch.linspace(0, decay_mod, tau).to(data.device)
    noisy_data = torch.zeros(tau, batch, pixels).to(data.device)
    noisy_data[0] = data

    for it in range(1, tau):
        noisy_data[it] = torch.normal(
            mean=noisy_data[it - 1], std=noise_weighting[it]
        ).clamp(0, 1)
    noisy_data = rearrange(noisy_data, "tau batch pixels -> (batch tau) pixels")
    return noisy_data


def add_normal_noise_multiple(data, tau, decay_mod=1.0):
    """
    Distorting the data by sampling from one normal distribution with a fixed mean.
    Adding the noise with different weights to the tensor.
    """
    if data.dim() == 1:
        data = data.unsqueeze(0)
    batch, pixels = data.shape
    noise = torch.normal(mean=0.5, std=0.2, size=(batch, pixels)).to(
        data.device
    )  # normal distribution
    data_expanded = repeat(data, "batch pixels -> tau batch pixels", tau=tau)
    noise_expanded = repeat(noise, "batch pixels -> tau batch pixels", tau=tau)
    noise_weighting = torch.linspace(0, 1, tau).to(data.device) ** decay_mod
    noise_weighting = noise_weighting / noise_weighting.max()  # normalize
    noise_weighting = repeat(noise_weighting, "tau -> tau batch 1", batch=batch)
    noisy_data = (
        data_expanded * (1 - noise_weighting) + noise_expanded * noise_weighting
    )
    noisy_data = noisy_data.clamp(0, 1)
    noisy_data = rearrange(noisy_data, "tau batch pixels -> (batch tau) pixels")
    return noisy_data
