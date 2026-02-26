import time
import torch
from src.metrics.snr_metric import SNRMetric
from src.metrics.lsd_metric import LSDMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch processing, logging, and evaluation.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        wav = batch["data_object"]
        wav_l = batch["wav_l"]
        band = batch["band"]


        if self.is_train:
            self.optimizer.zero_grad()

        loss, weight = self.criterion(
            net=self.model,
            y=wav,
            wav_l=wav_l,
            band=band
        )

        # loss = self.criterion(est_noise, z)["loss"]
        batch["loss"] = loss
        if self.is_train:
            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        metrics.update("loss", loss.item())
        #metrics.update("weight", weight[0].item())
        #self.train_metrics.update("weight", weight[0].item())

        return batch
    
    def _log_batch(self, batch_idx, batch, mode="train"):
        prefix = f"{mode}/"
        # log whatever is in batch that the tracker knows about
        for key in ["loss"]:
            self.writer.add_scalar(key, batch[key])