import torch
from src.metrics.base_metric import BaseMetric

class LSDMetric(BaseMetric):
    """
    Compute per-batch Log Spectral Distortion (LSD) over the full spectrum:
      sqrt(mean((log|STFT(pred)|^2 - log|STFT(target)|^2)^2)).
    """

    def __init__(self, device: str = "auto", name: str = None):
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> float:
        # move to correct device
        preds = predictions.to(self.device)
        targs = targets.to(self.device)

        # compute magnitude STFT
        def mag(x):
            c = torch.view_as_real(torch.stft(
                x,
                n_fft=2048,
                hop_length=512,
                window=torch.hann_window(2048, device=self.device),
                return_complex=True,
            ))
            return torch.norm(c, p=2, dim=-1)  # [B, F, T]

        # log-power spectra
        sp = torch.log10(mag(preds).square().clamp(min=1e-8))
        st = torch.log10(mag(targs).square().clamp(min=1e-8))

        # full-band LSD
        lsd = (sp - st).square().mean(dim=1).sqrt().mean()
        return lsd.item()
