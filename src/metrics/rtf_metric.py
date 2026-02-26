import torch
from src.metrics.base_metric import BaseMetric

class RTFMetric(BaseMetric):
    """
    Real-Time-Factor = processing_time (s) / audio_duration (s).
    """
    def __init__(self, name: str = "RTF"):
        super().__init__(name=name)

    def __call__(self,
                 predictions: torch.Tensor,
                 targets:      torch.Tensor,
                 processing_time: float,
                 sampling_rate:   int,
                 **kwargs
                ) -> float:
        # targets.shape[-1] is number of samples
        duration = targets.shape[-1] / float(sampling_rate)
        return processing_time / (duration + 1e-8)
