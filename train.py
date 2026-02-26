import warnings
import os
from copy import deepcopy

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


class EMACallback(Callback):
    def __init__(self, filepath, alpha=0.999, k=3):
        super().__init__()
        self.alpha = alpha
        self.filepath = filepath
        self.k = k
        self.queue = []
        self.last_parameters = None

    @rank_zero_only
    def _del_model(self, epoch_to_remove):
        fp = self.filepath.format(epoch=epoch_to_remove)
        if os.path.exists(fp):
            os.remove(fp)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.last_parameters = getattr(self, 'current_parameters', deepcopy(pl_module.state_dict()))

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.current_parameters = deepcopy(pl_module.state_dict())
        for k, v in self.current_parameters.items():
            v.copy_(self.alpha * v + (1.0 - self.alpha) * self.last_parameters[k])
        del self.last_parameters

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        if hasattr(self, 'current_parameters'):
            epoch = trainer.current_epoch
            self.queue.append(epoch)
            save_path = self.filepath.format(epoch=epoch)
            torch.save(self.current_parameters, save_path)
            pl_module.print(f'{save_path} is saved')
            while len(self.queue) > self.k:
                self._del_model(self.queue.pop(0))
        else:
            self.current_parameters = deepcopy(pl_module.state_dict())


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main training script.
    """
    set_random_seed(config.trainer.seed)
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    # --- Device setup ---
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # --- Load dataloaders ---
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # --- Instantiate model ---
    model = instantiate(config.model).to(device)
    logger.info(model)

    # --- FLOPs and Params ---
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        example_batch = next(iter(dataloaders["train"]))
        audio = example_batch["data_object"].to(device)
        audio_low = example_batch["wav_l"].to(device)
        band = example_batch["band"].to(device).long()
        sigma = torch.ones(audio.size(0), 1, device=device)

        print(f"Example shapes → "
              f"audio: {tuple(audio.shape)}, "
              f"sigma: {tuple(sigma.shape)}, "
              f"audio_low: {tuple(audio_low.shape)}, "
              f"band: {tuple(band.shape)} (dtype={band.dtype})")

        # Use a tuple input matching model.forward signature
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (audio, sigma, audio_low, band))
            total_flops = flops.total() / 1e9  # GFLOPs

        param_table = parameter_count_table(model)
        logger.info(f"🔹 Model FLOPs: {total_flops:.3f} GFLOPs")
        logger.info("\n" + param_table)
        print(f"🔹 Model FLOPs: {total_flops:.3f} GFLOPs")
        print(param_table)

    except Exception as e:
        logger.warning(f"❌ FLOPs/MACs calculation failed: {e}")

    # --- Loss and metrics ---
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # --- Optimizer and scheduler ---
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=None,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
