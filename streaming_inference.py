import hydra
import librosa as rosa
import torch
import time
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import os
from glob import glob
from scipy.signal import sosfiltfilt, cheby1, resample_poly

from src.utils.init_utils import set_random_seed
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.trainer.inferencer import edm_sampler


class StreamingEDMSampler:
    def __init__(self, net, num_steps=8, rho=7, sigma_min=0.002, sigma_max=80,
                 guidance=1.0, gnet=None, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        self.net = net
        self.gnet = gnet
        self.num_steps = num_steps
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.guidance = guidance
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.slots = [None] * num_steps

    def _make_t_steps(self, device, dtype):
        step_indices = torch.arange(self.num_steps, device=device, dtype=dtype)
        t = (self.sigma_max ** (1 / self.rho) +
             step_indices / (self.num_steps - 1) *
             (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return torch.cat([t, torch.zeros(1, device=device, dtype=dtype)])

    @torch.inference_mode()
    def denoise(self, x, sigma, wav_l, band):
        denoised = self.net(x, sigma, wav_l, band)
        if self.guidance == 1.0 or self.gnet is None:
            return denoised
        denoised_ref = self.gnet(x, sigma, wav_l, band)
        return denoised_ref.lerp(denoised, self.guidance)

    @torch.inference_mode()
    def _one_euler_step(self, x, t_cur, t_next, wav_l, band):
        dtype = x.dtype
        device = x.device
        bsz = x.shape[0]
        t_hat = t_cur
        x_hat = x
        if self.S_churn > 0 and self.S_min <= t_cur.item() <= self.S_max:
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x)
        sigma_hat = t_hat * torch.ones(bsz, device=device, dtype=dtype)
        d = (x_hat - self.denoise(x_hat, sigma_hat, wav_l, band)) / t_hat
        return x_hat + (t_next - t_hat) * d

    @torch.inference_mode()
    def process_new_chunk(self, wav_l_chunk, band_chunk, device):
        dtype = wav_l_chunk.dtype
        t_steps = self._make_t_steps(device, dtype)
        finished_chunk = self.slots[self.num_steps - 1]
        new_slots = [None] * self.num_steps
        x_init = torch.randn_like(wav_l_chunk) * t_steps[0]
        new_slots[0] = {'x': x_init, 'step': 0, 'wav_l': wav_l_chunk, 'band': band_chunk}
        for i in range(self.num_steps - 1):
            slot = self.slots[i]
            if slot is None:
                new_slots[i + 1] = None
                continue
            step = slot['step']
            t_cur = t_steps[step]
            t_next = t_steps[step + 1]
            x_next = self._one_euler_step(slot['x'], t_cur, t_next, slot['wav_l'], slot['band'])
            new_slots[i + 1] = {'x': x_next, 'step': step + 1, 'wav_l': slot['wav_l'], 'band': slot['band']}
        self.slots = new_slots
        if finished_chunk is None:
            return None
        return finished_chunk['x'].clamp(-1, 1 - torch.finfo(torch.float32).eps).squeeze(0)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = hydra.utils.instantiate(config.model).to(device)
    bad_model = hydra.utils.instantiate(config.model).to(device) if config.inferencer.get("bad_model") else None

    if config.inferencer.get("from_pretrained"):
        ckpt = torch.load(f"saved/edm_convnetxt/{config.inferencer.from_pretrained}",
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    wav_files = glob("*.wav") + glob("*.WAV")
    if not wav_files:
        raise FileNotFoundError("No .wav file found in current directory!")
    input_path = wav_files[0]
    print(f"Using input file: {input_path}")

    wav, _ = rosa.load(input_path, sr=config.audio.sampling_rate)
    wav = wav / np.max(np.abs(wav))

    low_sr = config.audio.reduced_sampling_rate

    order = 8
    ripple = 0.05
    nyq = 0.5 * config.audio.sampling_rate

    highcut = low_sr / 2
    hi = highcut / nyq

    if hi < 1:
        sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
        wav_l = sosfiltfilt(sos, wav)
        wav_l = resample_poly(wav_l, low_sr, config.audio.sampling_rate)
        wav_l = resample_poly(wav_l, config.audio.sampling_rate, low_sr)
    else:
        wav_l = wav.copy()

    if len(wav_l) != len(wav):
        wav_l = wav_l[:len(wav)] if len(wav_l) > len(wav) else np.pad(wav_l, (0, len(wav) - len(wav_l)))

    fft_size = config.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1

    wav_hr = torch.from_numpy(wav).float().to(device)
    wav_l_full = torch.from_numpy(wav_l).float().to(device)
    band_full = band.to(device)

    sr = config.audio.sampling_rate

    stream_cfg = config.inferencer.streaming
    chunk_size = config.audio.length
    chunk_sec = chunk_size / sr
    hop_size = int(chunk_size * stream_cfg.overlap_ratio)
    use_overlap = stream_cfg.use_overlap
    num_flush = stream_cfg.num_flush_chunks
    max_test_chunks = stream_cfg.get("max_test_chunks", 999999)

    print(f"File length: {len(wav_hr)} samples | Chunk size: {chunk_size}")

    model.eval()

    def run_version(version_name: str, use_ol: bool, is_sequential: bool = False):
        if is_sequential:
            print(f"\n=== {version_name} (Sequential - full denoising per chunk) ===")
        else:
            print(f"\n=== {version_name} ===")

        sampler = StreamingEDMSampler(net=model, gnet=bad_model, num_steps=config.inferencer.steps,
                                      rho=config.inferencer.rho, sigma_min=config.inferencer.get("sigma_min", 0.002),
                                      sigma_max=config.inferencer.get("sigma_max", 80),
                                      guidance=config.inferencer.guidance,
                                      S_churn=config.inferencer.S_churn,
                                      S_min=config.inferencer.S_min,
                                      S_max=config.inferencer.S_max,
                                      S_noise=config.inferencer.S_noise)

        step = hop_size if use_ol else chunk_size
        real_chunks = []

        wav_l_src = wav_l_full
        if len(wav_l_src) < chunk_size:
            wav_l_src = torch.nn.functional.pad(wav_l_src, (0, chunk_size - len(wav_l_src)))

        for start in range(0, len(wav_l_src) - chunk_size + 1, step):
            real_chunks.append(wav_l_src[start:start + chunk_size].clone())
            if len(real_chunks) >= max_test_chunks and not is_sequential:
                break

        if len(real_chunks) == 0:
            real_chunks.append(wav_l_src[:chunk_size].clone())

        all_chunks = real_chunks + [real_chunks[-1]] * num_flush if not is_sequential else real_chunks

        latencies = []
        recon_list = []
        total_proc_time = 0.0

        for wav_l_chunk in tqdm(all_chunks, desc=f"Processing {version_name}"):
            t0 = time.time()
            wav_l_dev = wav_l_chunk.unsqueeze(0).to(device)
            band_dev = band_full.unsqueeze(0).to(device)

            if is_sequential:
                noise = torch.randn_like(wav_l_dev)
                hr_chunk = edm_sampler(net=model, x_init=noise, wav_l=wav_l_dev, band=band_dev, gnet=bad_model,
                                       num_steps=config.inferencer.steps, rho=config.inferencer.rho,
                                       sigma_min=config.inferencer.get("sigma_min", 0.002),
                                       sigma_max=config.inferencer.get("sigma_max", 80),
                                       guidance=config.inferencer.guidance,
                                       S_churn=config.inferencer.S_churn,
                                       S_min=config.inferencer.S_min,
                                       S_max=config.inferencer.S_max,
                                       S_noise=config.inferencer.S_noise)
                hr_chunk = hr_chunk.squeeze(0)
                torch.cuda.empty_cache()
            else:
                hr_chunk = sampler.process_new_chunk(wav_l_dev, band_dev, device)

            chunk_latency = time.time() - t0
            latencies.append(chunk_latency)
            total_proc_time += chunk_latency

            if hr_chunk is not None:
                recon_list.append(hr_chunk.cpu().detach().squeeze(0))

        if use_ol:
            window = torch.hann_window(chunk_size)
            ola_length = (len(real_chunks) + 1) * hop_size + chunk_size
            ola_buffer = torch.zeros(ola_length)
            weight_buffer = torch.zeros(ola_length)
            for idx, chunk in enumerate(recon_list):
                out_start = idx * hop_size
                end = min(out_start + chunk_size, ola_length)
                length = end - out_start
                chunk_cpu = chunk[:length] * window[:length]
                ola_buffer[out_start:end] += chunk_cpu
                weight_buffer[out_start:end] += window[:length]
            mask = weight_buffer > 1e-8
            ola_buffer[mask] /= weight_buffer[mask]
            recon_np = ola_buffer[:len(wav_hr)].numpy()
        else:
            recon_np = np.concatenate([c.numpy().flatten() for c in recon_list])[:len(wav_hr)]
            if len(recon_np) < len(wav_hr):
                recon_np = np.pad(recon_np, (0, len(wav_hr) - len(recon_np)), mode='constant')

        output_path = str(ROOT_PATH / "data" / f"reconstructed_{version_name.lower()}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, recon_np, sr)

        avg_latency = np.mean(latencies) * 1000
        throughput = len(recon_list) / total_proc_time if total_proc_time > 0 else 0
        jitter = np.std(latencies) * 1000

        print(f"Latency (avg):     {avg_latency:6.1f} ms")
        print(f"Throughput:        {throughput:6.2f} chunks/sec")
        print(f"Jitter:            {jitter:6.1f} ms")

        recon_tensor = torch.from_numpy(recon_np).unsqueeze(0).to(device)
        metrics = hydra.utils.instantiate(config.metrics)
        tracker = MetricTracker(*[m.name for m in metrics["inference"]])
        for m in metrics["inference"]:
            kwargs = {}
            if m.name == "RTF":
                kwargs = {"processing_time": total_proc_time, "sampling_rate": sr}
            elif m.name in ("EVAL_LSDLF", "EVAL_LSDHF"):
                hf_bin = int(1025 * (config.audio.reduced_sampling_rate / sr))
                kwargs = {"hf_bin": hf_bin}
            val = m(predictions=recon_tensor, targets=wav_hr.unsqueeze(0), **kwargs)
            tracker.update(m.name, val.item() if torch.is_tensor(val) else val)

        print("Quality metrics:")
        result = tracker.result()
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        print(f"  {key}.{sub_key}: {sub_val:.4f}")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    run_version("WITH_OVERLAP", use_ol=True, is_sequential=False)
    run_version("WITHOUT_OVERLAP", use_ol=False, is_sequential=False)
    run_version("SEQUENTIAL", use_ol=False, is_sequential=True)
    run_version("SEQUENTIAL_WITH_OVERLAP", use_ol=True, is_sequential=True)


if __name__ == "__main__":
    main()
