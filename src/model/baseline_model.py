import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log

Linear = nn.Linear
silu = F.silu
relu = F.relu

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class GlobalResponseNorm(nn.Module):
    """GRN from ConvNeXt v2 - minimal FLOPS overhead, big impact"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, C, T)
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D signals (B, C, T)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, T)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_channels = hparams.dpm.pos_emb_channels
        self.linear_scale = hparams.dpm.pos_emb_scale
        self.out_channels = hparams.arch.pos_emb_dim

        self.projection1 = Linear(self.n_channels, self.out_channels)
        self.projection2 = Linear(self.out_channels, self.out_channels)

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        emb = log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32).to(noise_level) * -emb)
        emb = self.linear_scale * noise_level.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.projection1(emb)
        emb = silu(emb)
        emb = self.projection2(emb)
        emb = silu(emb)
        return emb


class BSFT(nn.Module):
    """Optimized BSFT with GRN"""
    def __init__(self, nhidden, out_channels, use_depthwise=True):
        super().__init__()
        if use_depthwise:
            self.mlp_shared = nn.Sequential(
                nn.Conv1d(2, 2, kernel_size=3, padding=1, groups=2),
                nn.Conv1d(2, nhidden, kernel_size=1)
            )
        else:
            self.mlp_shared = nn.Conv1d(2, nhidden, kernel_size=3, padding=1)

        self.mlp_gamma = Conv1d(nhidden, out_channels, kernel_size=1)
        self.mlp_beta = Conv1d(nhidden, out_channels, kernel_size=1)
        
        # Add GRN for better feature normalization
        self.grn = GlobalResponseNorm(nhidden)

    def forward(self, x, band):
        actv = self.mlp_shared(band)
        actv = self.grn(actv)
        actv = silu(actv)
        
        gamma = self.mlp_gamma(actv).unsqueeze(-1)
        beta = self.mlp_beta(actv).unsqueeze(-1)
        return x * (1 + gamma) + beta


class EfficientFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, filter_length=1024, 
                 hop_length=256, win_length=1024, sampling_rate=48000, use_checkpoint=False):
        super(EfficientFourierUnit, self).__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.use_checkpoint = use_checkpoint
        
        hann_window = torch.hann_window(win_length)
        self.register_buffer('hann_window', hann_window)

        self.conv_layer = Conv2d(in_channels * 2, out_channels * 2,
                                 kernel_size=1, padding=0, bias=False)
        self.bsft = BSFT(bsft_channels, out_channels * 2, use_depthwise=True)

    def forward(self, x, band):
        batch = x.shape[0]
        x = x.view(-1, x.size()[-1])

        ffted = torch.stft(x, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, 
                          window=self.hann_window, center=True, normalized=True, 
                          onesided=True, return_complex=False)
        ffted = ffted.permute(0, 3, 1, 2).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[2:])

        ffted = relu(self.bsft(ffted, band))
        ffted = self.conv_layer(ffted)

        ffted = ffted.view((-1, 2,) + ffted.size()[2:]).permute(0, 2, 3, 1).contiguous()
        output = torch.istft(torch.view_as_complex(ffted), self.n_fft, 
                           hop_length=self.hop_size, win_length=self.win_size, 
                           window=self.hann_window, center=True, normalized=True, 
                           onesided=True)
        output = output.view(batch, -1, x.size()[-1])
        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, use_checkpoint=False, **audio_kwargs):
        super(SpectralTransform, self).__init__()
        self.conv1 = Conv1d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.fu = EfficientFourierUnit(out_channels // 2, out_channels // 2, bsft_channels, 
                                       use_checkpoint=use_checkpoint, **audio_kwargs)
        self.conv2 = Conv1d(out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x, band):
        x = silu(self.conv1(x))
        output = self.fu(x, band)
        output = self.conv2(x + output)
        return output


class ConvNeXtV2Block(nn.Module):
    """
    1. Depthwise conv with large kernel (7x7 -> 7 for 1D)
    2. LayerNorm instead of BatchNorm
    3. Inverted bottleneck (expand then contract)
    4. GRN for better channel interactions
    5. GELU activation
    """
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6, mlp_ratio=4):
        super().__init__()
        # Depthwise conv with larger kernel
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1d(dim)
        
        # Inverted bottleneck MLP
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(hidden_dim)  # Key innovation from v2
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, 1)
        
        # Layer scale (helps training stability)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = nn.Identity()  # Can add DropPath if needed

    def forward(self, x):
        # x: (B, C, T)
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma[:, None] * x
        
        x = shortcut + self.drop_path(x)
        return x


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, bsft_channels, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, padding=1, use_checkpoint=False,
                 use_convnext=False, **audio_kwargs):
        super(FFC, self).__init__()

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        self.use_convnext = use_convnext

        if use_convnext and in_cl > 0:
            # Use ConvNeXt block for local processing
            self.convl2l = nn.Sequential(
                Conv1d(in_cl, out_cl, 1, bias=False),
                ConvNeXtV2Block(out_cl, mlp_ratio=2)
            )
        else:
            self.convl2l = nn.Sequential(
                nn.Conv1d(in_cl, in_cl, kernel_size, padding=padding, groups=in_cl, bias=False),
                nn.Conv1d(in_cl, out_cl, 1, bias=False)
            ) if in_cl > 0 else nn.Identity()
        
        self.convl2g = Conv1d(in_cl, out_cg, 1, bias=False) if in_cl > 0 else None
        self.convg2l = Conv1d(in_cg, out_cl, 1, bias=False) if in_cg > 0 else None
        self.convg2g = SpectralTransform(in_cg, out_cg, bsft_channels, 
                                        use_checkpoint=use_checkpoint, **audio_kwargs) if in_cg > 0 else None

    def forward(self, x_l, x_g, band):
        out_xl = self.convl2l(x_l)
        if self.convg2l is not None:
            out_xl = out_xl + self.convg2l(x_g)
            
        out_xg = torch.zeros_like(x_g) if self.convl2g is None else self.convl2g(x_l)
        if self.convg2g is not None:
            out_xg = out_xg + self.convg2g(x_g, band)

        return out_xl, out_xg


class ResidualBlock(nn.Module):
    """Optimized Residual Block with ConvNeXt v2 features"""
    def __init__(self, residual_channels, pos_emb_dim, bsft_channels, 
                 use_checkpoint=False, use_convnext=False, **audio_kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.ffc1 = FFC(residual_channels, 2*residual_channels, bsft_channels,
                       kernel_size=3, ratio_gin=0.5, ratio_gout=0.5, padding=1, 
                       use_checkpoint=use_checkpoint, use_convnext=use_convnext,
                       **audio_kwargs)

        self.diffusion_projection = Linear(pos_emb_dim, residual_channels)
        
        # Add GRN before output projection
        self.grn = GlobalResponseNorm(residual_channels)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def _forward(self, x, band, noise_level):
        noise_level = self.diffusion_projection(noise_level).unsqueeze(-1)
        y = x + noise_level
        y_l, y_g = torch.split(y, [y.shape[1] - self.ffc1.global_in_num, self.ffc1.global_in_num], dim=1)
        y_l, y_g = self.ffc1(y_l, y_g, band)
        
        gate_l, filter_l = torch.chunk(y_l, 2, dim=1)
        gate_g, filter_g = torch.chunk(y_g, 2, dim=1)
        gate = torch.cat((gate_l, gate_g), dim=1)
        filter = torch.cat((filter_l, filter_g), dim=1)
        
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.grn(y)  # Apply GRN before projection
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
    
    def forward(self, x, band, noise_level):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, band, noise_level, use_reentrant=False)
        return self._forward(x, band, noise_level)


class NuWave2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_projection = Conv1d(2, hparams.arch.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(hparams)
        
        audio_kwargs = dict(
            filter_length=hparams.audio.filter_length,
            hop_length=hparams.audio.hop_length,
            win_length=hparams.audio.win_length,
            sampling_rate=hparams.audio.sampling_rate,
        )
        
        use_checkpoint = getattr(hparams.arch, 'use_checkpoint', False)
        use_convnext = getattr(hparams.arch, 'use_convnext', False)
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                hparams.arch.residual_channels,
                hparams.arch.pos_emb_dim,
                hparams.arch.bsft_channels,
                use_checkpoint=use_checkpoint,
                use_convnext=use_convnext,
                **audio_kwargs,
            )
            for _ in range(hparams.arch.residual_layers)
        ])
        self.len_res = len(self.residual_layers)
        self.skip_projection = Conv1d(hparams.arch.residual_channels, 
                                      hparams.arch.residual_channels, 1)
        self.output_projection = Conv1d(hparams.arch.residual_channels, 1, 1)

    def forward(self, audio, audio_low, band, noise_level):
        x = torch.stack((audio, audio_low), dim=1)
        x = self.input_projection(x)
        x = silu(x)
        noise_emb = self.diffusion_embedding(noise_level)
        band_onehot = F.one_hot(band).transpose(1, -1).float()

        skip = 0.0
        for layer in self.residual_layers:
            x, skip_connection = layer(x, band_onehot, noise_emb)
            skip += skip_connection

        x = skip / sqrt(self.len_res)
        x = self.skip_projection(x)
        x = silu(x)
        return self.output_projection(x).squeeze(1)


class Diffusion(nn.Module):
    """
    Wraps NuWave2 with forward diffusion, noise scheduling,
    and denoising. Exposes both .diffusion() and .forward() methods.
    """
    def __init__(self, hparams):
        super().__init__()
        self.model = NuWave2(hparams)
        self.logsnr_min = hparams.logsnr.logsnr_min
        self.logsnr_max = hparams.logsnr.logsnr_max
        self.logsnr_b = atan(exp(-self.logsnr_max / 2))
        self.logsnr_a = atan(exp(-self.logsnr_min / 2)) - self.logsnr_b

    def snr(self, t: torch.Tensor):
        logsnr = -2 * torch.log(torch.tan(self.logsnr_a * t + self.logsnr_b))
        norm_nlogsnr = (
            self.logsnr_max - logsnr
        ) / (self.logsnr_max - self.logsnr_min)
        alpha_sq = torch.sigmoid(logsnr)
        sigma_sq = torch.sigmoid(-logsnr)
        return logsnr, norm_nlogsnr, alpha_sq, sigma_sq

    def diffusion(
        self, signal: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward SDE step: returns (alpha, sigma, noisy_signal)
        """
        _, norm_nlogsnr, alpha_sq, sigma_sq = self.snr(t)
        alpha = torch.sqrt(alpha_sq).unsqueeze(-1)
        sigma = torch.sqrt(sigma_sq).unsqueeze(-1)
        noised = alpha * signal + sigma * noise
        return alpha, sigma, noised

    def forward(
        self,
        noised: torch.Tensor,
        audio_low: torch.Tensor,
        band: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Estimate the noise given a noised signal. If `z` is provided,
        skips sampling.
        Returns (est_noise, logsnr, (alpha_sq, sigma_sq)).
        """
        logsnr, norm_nlogsnr, alpha_sq, sigma_sq = self.snr(t)
        # noise arg unused here since we only estimate
        est_noise = self.model(noised, audio_low, band, norm_nlogsnr)
        return est_noise, logsnr, (alpha_sq, sigma_sq)
    
    def denoise(self, y, y_l, band, t, h):
        noise, logsnr_t, (alpha_sq_t, sigma_sq_t) = self(y, y_l, band, t)
        
        f_t = - self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)
        g_t_sq = 2 * self.logsnr_a * torch.tan(self.logsnr_a * t + self.logsnr_b)

        dzt_det = (f_t * y - 0.5 * g_t_sq * (-noise / torch.sqrt(sigma_sq_t))) * h

        denoised = y - dzt_det
        return denoised

    def denoise_ddim(self, y, y_l, band, logsnr_t, logsnr_s, z=None):
        """
        Denoising step for DDIM. Calculates denoising given t and s logsnr.
        """
        norm_nlogsnr = (self.logsnr_max - logsnr_t) / (self.logsnr_max - self.logsnr_min)
        alpha_sq_t, sigma_sq_t = torch.sigmoid(logsnr_t), torch.sigmoid(-logsnr_t)
    
        if z is None:
            noise = self.model(y, y_l, band, norm_nlogsnr)
        else:
            noise = z
    
        alpha_sq_s, sigma_sq_s = torch.sigmoid(logsnr_s), torch.sigmoid(-logsnr_s)

        sqrt_sigma_sq_t = torch.sqrt(sigma_sq_t).unsqueeze(1)  # (batch, 1)
        sqrt_alpha_sq_t = torch.sqrt(alpha_sq_t).unsqueeze(1)    # (batch, 1)
    
        pred = (y - sqrt_sigma_sq_t * noise) / sqrt_alpha_sq_t
    
        denoised = torch.sqrt(alpha_sq_s).unsqueeze(1) * pred + torch.sqrt(sigma_sq_s).unsqueeze(1) * noise
        return denoised, pred

class EDMPrecond(nn.Module):
    def __init__(self, hparams, sigma_data=0.5):
        super().__init__()
        self.model = NuWave2(hparams)
        self.sigma_data = sigma_data

    def forward(self, x, sigma, audio_low, band):
        sigma_in = sigma.to(x.dtype)
        sigma_b = sigma_in

        # compute preconditioning coefficients (Eq.7)
        c_skip = self.sigma_data**2 / (sigma_b**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma_b**2 + self.sigma_data**2)
        c_out = sigma_b * self.sigma_data / torch.sqrt(sigma_b**2 + self.sigma_data**2)


        if not self.training:
            c_skip = c_skip[:, None]
            c_in = c_in[:, None]
            c_out = c_out[:, None]

        c_noise = sigma_in.log() / 4.0
        x_scaled = c_in * x  # correct broadcasting in both train/inference

        F_x = self.model(
            x_scaled,
            audio_low,
            band,
            c_noise,
        )

        return c_skip * x + c_out * F_x

