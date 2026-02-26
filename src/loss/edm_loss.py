import torch
from torch import nn


class EDMLoss(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def forward(self, net, y, wav_l, band):
        B = y.shape[0]
        # sample sigma
        rnd = torch.randn(y.shape[0], device=y.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        # weight w(sigma)
        weight = (sigma**2 + self.sigma_data**2) / (sigma*self.sigma_data)**2
        # noisy input
        expand_shape = [B] + [1] * (y.dim() - 1)
        sigma = sigma.view(*expand_shape)
        weight = weight.view(*expand_shape)

        n = torch.randn_like(y) * sigma
        # denoiser
        D_yn = net(y + n, sigma, wav_l, band)
        
        return ((D_yn - y).pow(2)).mean(), weight

