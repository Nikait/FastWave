import torch
from src.metrics.base_metric import BaseMetric

class SNRMetric(BaseMetric):
    """
    Compute per‐batch Signal‐to‐Noise‐Ratio:
      10 * log10( sum(target^2) / sum((target - pred)^2) ).
    """

    def __init__(self, device: str = "auto", name: str = None):
        """
        Args:
            device (str): "auto" | "cpu" | "cuda"
            name  (str): Optional override for metric name.
        """
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def __call__(self,
                 predictions: torch.Tensor,
                 targets:      torch.Tensor,
                 **kwargs) -> float:
        # move to correct device
        preds = predictions.to(self.device)
        targs = targets.to(self.device)

        # compute noise
        noise = targs - preds

        # signal & noise power
        power_signal = torch.sum(targs ** 2)
        power_noise  = torch.sum(noise ** 2)

        # avoid div‑zero
        snr_val = 10.0 * torch.log10(power_signal / (power_noise + 1e-8))

        # return as Python float
        return snr_val.item()
