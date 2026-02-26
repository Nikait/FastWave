from abc import abstractmethod

import torch
from torch import nn
import time
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import librosa as rosa
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    #x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)  #[B, F, TT,2]
        mag = stft.abs() #[B, F, TT]
        return mag

class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        self._stft_mag = STFTMag()

        # define dataloaders
        self.train_dataloader = dataloaders["train"]

        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len
        
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        resume_from = config.trainer.epoch_len
        
        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )
            print("checking for checkpoint saving")
            
            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    @torch.inference_mode()
    def process_batch_inference(self, batch, metrics):
        from src.utils.schedule_utils import make_schedule
        from src.trainer.inferencer import edm_sampler

        # 1) move & transform
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        wav_orig = batch['data_object']
        wav_l = batch['wav_l']
        band = batch['band']
        
        # initial noise
        sig_shape = wav_l.shape
        noise = torch.randn_like(wav_orig)

        # run EDM sampler
        start = time.time()
        y = edm_sampler(
            net=self.model,
            x_init=noise,
            num_steps=8,
            rho=8,
            wav_l=wav_l,
            band=band,
            gnet=None,
            guidance=1.6,
            S_churn=0,
            S_min=0,
            S_max=0,
            S_noise=0,
        )
        end = time.time()
        processing_time = end - start

        wav_recon = y.clamp(-1, 1 - torch.finfo(y.dtype).eps)
        batch['prediction'] = wav_recon

        sr = self.config.audio.sampling_rate
        sr_reduced = self.config.audio.reduced_sampling_rate
        hf_bin = int(1025 * (sr_reduced / sr))

        for m in self.metrics["inference"]:
            kwargs = {}
            if m.name == "RTF":
                kwargs = {
                    "processing_time": processing_time,
                    "sampling_rate": sr
                }
                
            elif m.name in ("EVAL_LSDLF", "EVAL_LSDHF"):
                kwargs = {
                    "hf_bin": hf_bin
                }

            val = m(
                predictions=batch["prediction"],
                targets=batch["data_object"],
                **kwargs
            )
            batch[m.name] = val
            metrics.update(m.name, val)

        return batch
    
    @torch.inference_mode()
    def _evaluation_epoch(self, epoch, part, dataloader):
        from src.utils.schedule_utils import make_schedule

        # set eval mode & reset metrics
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()


        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=part)):
                # sample & metric update
                #print("eval batch: ", batch)
                batch = self.process_batch_inference(batch, self.evaluation_metrics)
                
                # for the very first batch, log audio+spectrograms
                if batch_idx == 0:
                    sr = int(self.config.audio.sampling_rate)
                    #print("batch: ")
                    #print(batch)
                    ref_np  = batch['wav_l'].cpu().numpy()
                    pred_np = batch['prediction'].cpu().numpy()

                    for i in range(ref_np.shape[0]):
                        # log audio
                        for label, arr in [('reference', ref_np[i]), ('predicted', pred_np[i])]:
                            key = f"{part}/epoch_{epoch}/{label}_audio_{i}"
                            self.writer.add_audio(
                                audio=arr,
                                sample_rate=sr,
                                step=epoch,
                                name=key
                            )

                        mag_ref = self._stft_mag(torch.from_numpy(ref_np[i])[None]).squeeze(0).numpy()
                        db_ref  = rosa.amplitude_to_db(mag_ref, ref=np.max, top_db=80.)
                        mag_pred= self._stft_mag(torch.from_numpy(pred_np[i])[None]).squeeze(0).numpy()
                        db_pred = rosa.amplitude_to_db(mag_pred, ref=np.max, top_db=80.)
                        for label, db in [('ref', db_ref), ('pred', db_pred)]:
                            img_key = f"{part}/epoch_{epoch}/{label}_spectrogram_{i}"
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.imshow(db, origin='lower', aspect='auto')
                            ax.set_title(f"Epoch {epoch} {label.upper()} Spectrogram {i}")
                            ax.set_xlabel('Frames')
                            ax.set_ylabel('Frequency Bin')
                            plt.tight_layout()
                            self.writer.add_image(name=img_key, image=fig, step=epoch)
                            plt.close(fig)

                # log scalar metrics each batch
                self.writer.set_step((epoch - 1) * len(dataloader) + batch_idx, "validation")
                self._log_scalars(self.evaluation_metrics)

        # aggregate and return
        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
    
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=len(self.train_dataloader))
        ):

            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * len(self.train_dataloader) + batch_idx)
                # self.logger.debug(
                #     "Train Epoch: {} {} Loss: {:.6f}".format(
                #         epoch, self._progress(batch_idx), batch["loss"].item()
                #     )
                # )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics
        
        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs


    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type) if self.batch_transforms else None
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        print(f"The checkpoint will be stored to {filename}")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device, weights_only=False)

        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path, attr="model"):
        """
        Init a specific submodule (e.g. "model" or "bad_model") from a pretrained .pth file.

        Args:
            pretrained_path (str): path to the model state dict.
            attr (str): name of the attribute on self to load (e.g. "model" or "bad_model").
        """
        pretrained_path = str(pretrained_path)

        # Log / print which attribute we're about to load into
        if hasattr(self, "logger"):
            self.logger.info(f"Loading weights for '{attr}' from: {pretrained_path} ...")
        else:
            print(f"Loading weights for '{attr}' from: {pretrained_path} ...")

        # Load the checkpoint
        checkpoint = torch.load(f"saved/edm_convnetxt/{pretrained_path}", map_location=self.device, weights_only=False)

        # Ensure that the attribute exists on self
        if not hasattr(self, attr):
            raise AttributeError(f"Cannot find attribute '{attr}' on {self.__class__.__name__} to load weights into.")

        target_module = getattr(self, attr)

        # If checkpoint wraps state_dict, unwrap it
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Finally load
        target_module.load_state_dict(state_dict)


