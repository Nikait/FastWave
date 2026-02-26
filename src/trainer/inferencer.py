import librosa as rosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import time
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.schedule_utils import make_schedule


def edm_sampler(
    net, x_init, wav_l, band, gnet=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, guidance=1.0,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    randn_like=torch.randn_like,
):
    dtype = x_init.dtype
    device = x_init.device

    # Guided denoiser
    def denoise(x, sigma):
        denoised_main = net(x, sigma, wav_l, band).to(dtype)
        if guidance == 1.0 or gnet is None:
            return denoised_main
        denoised_ref = gnet(x, sigma, wav_l, band).to(dtype)
        return denoised_ref.lerp(denoised_main, guidance)

    # Time step discretization
    step_indices = torch.arange(num_steps, device=device, dtype=dtype)
    t_steps = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Initial noise
    x_next = x_init.to(dtype) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step
        bsz = x_hat.shape[0]
        sigma_hat = t_hat * torch.ones(bsz, device=device, dtype=dtype)

        d_cur = (x_hat - denoise(x_hat, sigma_hat)) / sigma_hat[:, None]
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction (Heun)
        # if i < num_steps - 1:
        #     sigma_next = t_next * torch.ones(bsz, device=device, dtype=dtype)
        #     d_prime = (x_next - denoise(x_next, sigma_next)) / sigma_next[:, None]
        #     x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    @torch.no_grad()
    def forward(self, x):
        # x: [B, T] or [T]
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)
        mag = stft.abs()
        return mag


class Inferencer(BaseTrainer):
    """
    Inferencer (like Trainer but for inference).
    """

    def __init__(
        self,
        model,
        bad_model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.model = model
        self.bad_model = bad_model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        self.metrics = metrics
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics["inference"]], writer=None
        )

        self.steps = 8
        self.schedule = make_schedule(
            num_steps=self.steps,
            lambda_start=self.config.logsnr.logsnr_max,
            lambda_end=self.config.logsnr.logsnr_min,
        )

        self.S_churn = self.config.inferencer.S_churn
        self.S_min = self.config.inferencer.S_min
        self.S_max = self.config.inferencer.S_max
        self.S_noise = self.config.inferencer.S_noise

        self.stftmag = STFTMag().to(self.device)

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"), "model")

    def run_inference(self):
        self.model.eval()
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    @torch.inference_mode()
    def plot_spectrogram_to_numpy(self, y, y_recon, downsampled, save_path):
        fig = plt.figure(figsize=(9, 18))
        name_list = ['y0', 'downsampled', 'y_recon']
        for i, yy in enumerate([y, downsampled, y_recon]):
            yy = torch.clamp(yy, -1, 1 - 1e-6)
            # Move to CPU before numpy conversion
            mag = self.stftmag(yy.to(self.device)).cpu().numpy()
            ax = plt.subplot(6, 1, i + 1)
            ax.set_title(name_list[i])
            plt.imshow(
                rosa.amplitude_to_db(mag, ref=np.max, top_db=80.),
                vmax=0.0,
                aspect='auto',
                origin='lower',
                interpolation='none'
            )
            plt.xlabel('Frames')
            plt.ylabel('Channels')
            plt.tight_layout()

        fig.canvas.draw()
        fig.savefig(save_path, bbox_inches='tight')
        plt.close()

    @torch.inference_mode()
    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        wav_l = batch['wav_l']
        band = batch['band']
        noise = torch.randn_like(wav_l)

        start = time.time()
        y = edm_sampler(
            net=self.model,
            x_init=noise,
            num_steps=self.config.inferencer.steps,
            rho=self.config.inferencer.rho,
            wav_l=wav_l,
            band=band,
            gnet=self.bad_model,
            guidance=self.config.inferencer.guidance,
            S_churn=self.S_churn,
            S_min=self.S_min,
            S_max=self.S_max,
            S_noise=self.S_noise,
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

        if self.save_path is not None:
            out_dir = self.save_path / part
            out_dir.mkdir(exist_ok=True, parents=True)

            for i, (data_obj, pred, wav_l_sample) in enumerate(zip(
                    batch["data_object"],
                    wav_recon,
                    batch["wav_l"]
                )):

                # Save predicted tensor (raw)
                #torch.save(pred.cpu(), out_dir / f"pred_{batch_idx}_{i}.pt")

                # Save audio file (.wav)
                wav_path = out_dir / f"pred_{batch_idx}_{i}.wav"
                #sf.write(wav_path, pred.cpu().numpy(), sr)
                #print(f"[INFO] Saved reconstructed audio: {wav_path}")

                # Save spectrogram
                #self.plot_spectrogram_to_numpy(
                #    data_obj, pred, wav_l_sample,
                #    save_path=out_dir / f"pred_{batch_idx}_{i}.png"
                #)

        return batch


    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=part)):
            self.process_batch(
                batch_idx=batch_idx,
                batch=batch,
                metrics=self.evaluation_metrics,
                part=part,
            )

        return self.evaluation_metrics.result()


