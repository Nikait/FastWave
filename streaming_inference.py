import os
import time
import random
from glob import glob

import hydra
import librosa as rosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.utils.init_utils import set_random_seed
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.trainer.inferencer import edm_sampler


class StreamingEDMSampler:
    """
    Оставлен для совместимости, но в текущем скрипте не используется.
    """
    def __init__(
        self,
        net,
        num_steps=8,
        rho=7,
        sigma_min=0.002,
        sigma_max=80,
        guidance=1.0,
        gnet=None,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
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
        t = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1)
            * (
                self.sigma_min ** (1 / self.rho)
                - self.sigma_max ** (1 / self.rho)
            )
        ) ** self.rho
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
        new_slots[0] = {
            "x": x_init,
            "step": 0,
            "wav_l": wav_l_chunk,
            "band": band_chunk,
        }

        for i in range(self.num_steps - 1):
            slot = self.slots[i]
            if slot is None:
                new_slots[i + 1] = None
                continue
            step = slot["step"]
            t_cur = t_steps[step]
            t_next = t_steps[step + 1]
            x_next = self._one_euler_step(
                slot["x"], t_cur, t_next, slot["wav_l"], slot["band"]
            )
            new_slots[i + 1] = {
                "x": x_next,
                "step": step + 1,
                "wav_l": slot["wav_l"],
                "band": slot["band"],
            }

        self.slots = new_slots
        if finished_chunk is None:
            return None
        return finished_chunk["x"].clamp(
            -1, 1 - torch.finfo(torch.float32).eps
        ).squeeze(0)


def align_chunk_size(chunk_size: int, hop_length: int) -> int:
    aligned = (chunk_size // hop_length) * hop_length
    if aligned == 0:
        aligned = hop_length
    return aligned


def resample_exact(wav: np.ndarray, orig_sr: int, target_sr: int, target_len: int = None) -> np.ndarray:
    if orig_sr == target_sr:
        out = wav.astype(np.float32, copy=False)
        if target_len is not None:
            if len(out) > target_len:
                out = out[:target_len]
            elif len(out) < target_len:
                out = np.pad(out, (0, target_len - len(out)))
        return out

    out = resample_poly(wav, target_sr, orig_sr).astype(np.float32)

    if target_len is None:
        target_len = int(round(len(wav) * target_sr / orig_sr))

    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))

    return out


def make_low_quality_condition(wav_hr: np.ndarray, output_sr: int, input_sr: int) -> np.ndarray:
    """
    Правильная логика для этой модели:
    hr@48k -> downsample до input_sr -> upsample обратно до output_sr.
    На вход модели идёт low-quality сигнал ТОЙ ЖЕ ДЛИНЫ в сэмплах, что и hr target.
    """
    wav_l = resample_exact(wav_hr, output_sr, input_sr)
    wav_l_up = resample_exact(wav_l, input_sr, output_sr, target_len=len(wav_hr))
    return wav_l_up


def build_regular_chunks(signal_1d: torch.Tensor, chunk_size: int, hop_size: int):
    """
    Режет 1D сигнал на регулярные чанки с паддингом хвоста.
    Возвращает:
      chunks, original_len, padded_len
    """
    assert signal_1d.dim() == 1, "signal_1d must be 1D"

    orig_len = signal_1d.shape[0]

    if orig_len <= chunk_size:
        padded = F.pad(signal_1d, (0, chunk_size - orig_len))
        return [padded.clone()], orig_len, chunk_size

    n_chunks = int(np.ceil((orig_len - chunk_size) / hop_size)) + 1
    padded_len = (n_chunks - 1) * hop_size + chunk_size

    if padded_len > orig_len:
        signal_1d = F.pad(signal_1d, (0, padded_len - orig_len))

    chunks = []
    for start in range(0, padded_len - chunk_size + 1, hop_size):
        chunks.append(signal_1d[start:start + chunk_size].clone())

    return chunks, orig_len, padded_len


def flatten_dict(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key))
        else:
            out[full_key] = v
    return out


def canonical_metric_name(name: str):
    s = str(name).lower()

    if "rtf" in s:
        return "rtf"
    if "snr" in s:
        return "snr"
    if "lsd" in s and "hf" in s:
        return "lsd_hf"
    if "lsd" in s and "lf" in s:
        return "lsd_lf"
    if "lsd" in s:
        return "lsd"

    return None


def compute_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred, target: [1, T] или [T]
    """
    if pred.dim() == 2:
        pred = pred.squeeze(0)
    if target.dim() == 2:
        target = target.squeeze(0)

    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[:min_len]
    target = target[:min_len]

    noise = target - pred
    target_power = torch.sum(target ** 2)
    noise_power = torch.sum(noise ** 2)

    snr = 10.0 * torch.log10((target_power + eps) / (noise_power + eps))
    return float(snr.item())


def compute_lsd_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    split_bin: int = None,
    eps: float = 1e-7,
):
    """
    Возвращает:
      lsd_full, lsd_lf, lsd_hf
    """
    if win_length is None:
        win_length = n_fft

    if pred.dim() == 2:
        pred = pred.squeeze(0)
    if target.dim() == 2:
        target = target.squeeze(0)

    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[:min_len]
    target = target[:min_len]

    device = pred.device
    window = torch.hann_window(win_length, device=device)

    spec_pred = torch.stft(
        pred,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    spec_tgt = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )

    log_pred = 20.0 * torch.log10(spec_pred.abs().clamp_min(eps))
    log_tgt = 20.0 * torch.log10(spec_tgt.abs().clamp_min(eps))
    diff = log_pred - log_tgt  # [F, Frames]

    lsd_full = torch.sqrt((diff ** 2).mean(dim=0)).mean().item()

    freq_bins = diff.shape[0]
    if split_bin is None:
        split_bin = freq_bins // 2

    split_bin = max(1, min(freq_bins - 1, int(split_bin)))

    diff_lf = diff[:split_bin]
    diff_hf = diff[split_bin:]

    lsd_lf = torch.sqrt((diff_lf ** 2).mean(dim=0)).mean().item() if diff_lf.numel() > 0 else 0.0
    lsd_hf = torch.sqrt((diff_hf ** 2).mean(dim=0)).mean().item() if diff_hf.numel() > 0 else 0.0

    return float(lsd_full), float(lsd_lf), float(lsd_hf)


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _cuda_sync_if_needed(device):
    if isinstance(device, str):
        is_cuda = device.startswith("cuda")
    else:
        is_cuda = torch.device(device).type == "cuda"
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_checkpoint_if_needed(model, ckpt_ref, device):
    if not ckpt_ref:
        return model

    ckpt_path = str(ckpt_ref)
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join("saved", "edm_convnetxt", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def process_single_file(
    input_path,
    model,
    bad_model,
    config,
    device,
    input_sr,
    output_sr,
    chunk_size,
    band_full,
    version_name,
    use_ol,
    stream_cfg,
    save_path=None,
):
    """
    ВАЖНО:
    - модель ожидает x_init и wav_l одинаковой длины
    - поэтому wav_l должен быть degraded-to-input_sr, но затем upsampled обратно до output_sr
    - chunking делаем в домене output_sr
    """

    # 1) Грузим target HR в output_sr
    wav_hr, _ = rosa.load(input_path, sr=output_sr, mono=True)
    wav_hr = wav_hr.astype(np.float32)
    wav_hr = wav_hr / (np.max(np.abs(wav_hr)) + 1e-8)

    # 2) Делаем low-quality condition той же длины в сэмплах, что и wav_hr
    wav_l_up = make_low_quality_condition(wav_hr, output_sr=output_sr, input_sr=input_sr)

    wav_hr_t_cpu = torch.from_numpy(wav_hr).float()
    wav_l_full_cpu = torch.from_numpy(wav_l_up).float()

    hop_length = int(getattr(config.audio, "hop_length", 256))

    if use_ol:
        hop_size = int(round(chunk_size * float(stream_cfg.overlap_ratio)))
        hop_size = max(hop_length, align_chunk_size(hop_size, hop_length))
        hop_size = min(hop_size, chunk_size)
    else:
        hop_size = chunk_size

    real_chunks, orig_len, padded_len = build_regular_chunks(
        wav_l_full_cpu, chunk_size=chunk_size, hop_size=hop_size
    )

    latencies = []
    recon_list = []
    total_proc_time = 0.0

    band_dev = band_full.unsqueeze(0).to(device)

    for wav_l_chunk_cpu in real_chunks:
        wav_l_dev = wav_l_chunk_cpu.unsqueeze(0).to(device)

        # Для этой модели длина noise должна совпадать с длиной wav_l
        noise = torch.randn_like(wav_l_dev)

        _cuda_sync_if_needed(device)
        t0 = time.time()

        with torch.inference_mode():
            hr_chunk = edm_sampler(
                net=model,
                x_init=noise,
                wav_l=wav_l_dev,
                band=band_dev,
                gnet=bad_model,
                num_steps=config.inferencer.steps,
                rho=config.inferencer.rho,
                sigma_min=config.inferencer.get("sigma_min", 0.002),
                sigma_max=config.inferencer.get("sigma_max", 80),
                guidance=config.inferencer.guidance,
                S_churn=config.inferencer.S_churn,
                S_min=config.inferencer.S_min,
                S_max=config.inferencer.S_max,
                S_noise=config.inferencer.S_noise,
            )

        _cuda_sync_if_needed(device)
        chunk_latency = time.time() - t0

        hr_chunk = hr_chunk.squeeze(0).detach().cpu().float()

        # На всякий случай приводим длину к chunk_size
        if hr_chunk.shape[-1] > chunk_size:
            hr_chunk = hr_chunk[:chunk_size]
        elif hr_chunk.shape[-1] < chunk_size:
            hr_chunk = F.pad(hr_chunk, (0, chunk_size - hr_chunk.shape[-1]))

        latencies.append(chunk_latency)
        total_proc_time += chunk_latency
        recon_list.append(hr_chunk)

    # 3) Склейка чанков
    if use_ol:
        window = torch.hann_window(chunk_size)

        ola_buffer = torch.zeros(padded_len)
        weight_buffer = torch.zeros(padded_len)

        for idx, chunk in enumerate(recon_list):
            out_start = idx * hop_size
            out_end = out_start + chunk_size

            weighted = chunk * window
            ola_buffer[out_start:out_end] += weighted
            weight_buffer[out_start:out_end] += window

        mask = weight_buffer > 1e-8
        ola_buffer[mask] /= weight_buffer[mask]
        recon = ola_buffer[:orig_len]
    else:
        recon = torch.cat(recon_list, dim=0)[:orig_len]

    recon_np = recon.numpy()

    # 4) Сохранение
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, recon_np, output_sr)

    # 5) Метрики
    recon_cpu = torch.from_numpy(recon_np).float().unsqueeze(0)
    target_cpu = wav_hr_t_cpu[:len(recon_np)].unsqueeze(0)

    min_len = min(recon_cpu.shape[-1], target_cpu.shape[-1])
    recon_cpu = recon_cpu[..., :min_len]
    target_cpu = target_cpu[..., :min_len]

    recon_tensor = recon_cpu.to(device)
    target_tensor = target_cpu.to(device)

    audio_duration_sec = float(min_len / output_sr)

    # split bin для LF/HF метрик считаем по фактическому cutoff = input_sr / 2
    # относительно Nyquist output_sr / 2
    metric_n_bins = int(getattr(config.audio, "filter_length", 2048) // 2 + 1)
    hf_bin = int(round((input_sr / output_sr) * metric_n_bins))
    hf_bin = max(1, min(metric_n_bins - 1, hf_bin))

    raw_result = {}

    try:
        metrics = hydra.utils.instantiate(config.metrics)
        metric_list = metrics["inference"] if isinstance(metrics, dict) else metrics["inference"]

        tracker = MetricTracker(*[m.name for m in metric_list])

        for m in metric_list:
            m_name = getattr(m, "name", m.__class__.__name__)
            m_canon = canonical_metric_name(m_name)

            try:
                kwargs = {}

                if m_canon == "rtf":
                    # Пытаемся разными способами, т.к. сигнатура может отличаться
                    val = None
                    tried = [
                        {"processing_time": total_proc_time, "audio_duration": audio_duration_sec},
                        {"processing_time": total_proc_time, "sampling_rate": output_sr},
                        {"processing_time": total_proc_time},
                    ]
                    last_exc = None
                    for kw in tried:
                        try:
                            val = m(predictions=recon_tensor, targets=target_tensor, **kw)
                            break
                        except Exception as e:
                            last_exc = e
                    if val is None:
                        raise last_exc if last_exc is not None else RuntimeError("RTF metric call failed")
                else:
                    if m_canon in ("lsd_hf", "lsd_lf"):
                        kwargs["hf_bin"] = hf_bin
                    val = m(predictions=recon_tensor, targets=target_tensor, **kwargs)

                tracker.update(m_name, val.item() if torch.is_tensor(val) else float(val))

            except Exception as e:
                print(f"[WARN] metric {m_name} failed for {os.path.basename(input_path)}: {e}")

        raw_result = tracker.result()

    except Exception as e:
        print(f"[WARN] failed to instantiate/calculate project metrics for {os.path.basename(input_path)}: {e}")
        raw_result = {}

    # 6) Канонизация имён project metrics
    flat_raw = flatten_dict(raw_result)
    canonical = {}

    for k, v in flat_raw.items():
        if isinstance(v, (int, float)):
            c = canonical_metric_name(k)
            if c is not None and c not in canonical:
                canonical[c] = float(v)

    # 7) Manual fallback — чтобы summary точно был осмысленным
    split_bin_manual = max(1, min(metric_n_bins - 1, hf_bin))
    manual_snr = compute_snr(recon_cpu, target_cpu)
    manual_lsd, manual_lsd_lf, manual_lsd_hf = compute_lsd_metrics(
        recon_cpu.squeeze(0),
        target_cpu.squeeze(0),
        n_fft=int(getattr(config.audio, "filter_length", 2048)),
        hop_length=int(getattr(config.audio, "hop_length", 512)),
        win_length=int(getattr(config.audio, "win_length", getattr(config.audio, "filter_length", 2048))),
        split_bin=split_bin_manual,
    )
    manual_rtf = total_proc_time / max(audio_duration_sec, 1e-8)

    canonical.setdefault("snr", manual_snr)
    canonical.setdefault("lsd", manual_lsd)
    canonical.setdefault("lsd_lf", manual_lsd_lf)
    canonical.setdefault("lsd_hf", manual_lsd_hf)
    canonical.setdefault("rtf", manual_rtf)

    return canonical, latencies, total_proc_time, audio_duration_sec


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)
    random.seed(config.inferencer.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir = config.inferencer.get("dataset_dir", ".")
    all_wav_files = sorted(
        glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
        + glob(os.path.join(dataset_dir, "**", "*.WAV"), recursive=True)
    )

    if not all_wav_files:
        raise FileNotFoundError(f"No .wav files found in {dataset_dir}")

    n_files = min(5, len(all_wav_files))
    selected_files = random.sample(all_wav_files, n_files)

    print(f"Total .wav files found: {len(all_wav_files)}")
    print(f"Using {len(selected_files)} files for evaluation")

    OUTPUT_SR = 48000
    TEST_INPUT_SRS = [8000, 24000]

    summary_data = {}

    # ВАЖНО:
    # chunk size должен считаться в domain output_sr/model_sr, а не input_sr
    # т.к. модель получает wav_l уже upsampled обратно до output_sr.
    orig_chunk_sec = config.audio.length / config.audio.sampling_rate
    chunk_size = align_chunk_size(
        int(round(orig_chunk_sec * OUTPUT_SR)),
        int(getattr(config.audio, "hop_length", 256)),
    )

    for input_sr in TEST_INPUT_SRS:
        print(f"\n{'=' * 85}")
        print(f"TESTING INPUT SR = {input_sr} Hz  →  OUTPUT SR = {OUTPUT_SR} Hz")
        print(f"{'=' * 85}")

        lr_equiv = int(round(chunk_size * input_sr / OUTPUT_SR))
        print(
            f"  chunk_size (model/output domain): {chunk_size} samples  "
            f"({chunk_size / OUTPUT_SR:.3f} s; equivalent low-rate duration = {lr_equiv} samples @ {input_sr} Hz)"
        )

        # Band conditioning:
        # маска valid low-band рассчитывается по фактическому bandwidth input_sr/2
        fft_size = int(config.audio.filter_length // 2 + 1)
        cutoff_ratio = min(max(input_sr / OUTPUT_SR, 0.0), 1.0)
        cutoff_bin = int(round(cutoff_ratio * fft_size))
        cutoff_bin = max(1, min(fft_size, cutoff_bin))

        band = torch.zeros(fft_size, dtype=torch.int64)
        band[:cutoff_bin] = 1
        band_full = band.to(device)

        # Модель
        model = hydra.utils.instantiate(config.model).to(device)
        model = load_checkpoint_if_needed(
            model,
            config.inferencer.get("from_pretrained"),
            device,
        )
        model.eval()

        # optional guidance model
        bad_model = None
        bad_model_ref = config.inferencer.get("bad_model")
        if isinstance(bad_model_ref, str) and bad_model_ref:
            bad_model = hydra.utils.instantiate(config.model).to(device)
            bad_model = load_checkpoint_if_needed(bad_model, bad_model_ref, device)
            bad_model.eval()
        elif bad_model_ref:
            print("[WARN] config.inferencer.bad_model is truthy but not a checkpoint path; guidance model is skipped.")

        stream_cfg = config.inferencer.streaming

        regimes = ["sequential_no_overlap", "sequential_overlap_50pct"]
        summary_data[input_sr] = {}

        for use_ol, reg_name in zip([False, True], regimes):
            print(f"\n--- Running: {reg_name} ---")

            metrics_accum = {
                "snr": [],
                "lsd": [],
                "lsd_hf": [],
                "lsd_lf": [],
                "rtf": [],
            }
            all_latencies = []
            all_proc_times = []
            all_audio_durations = []

            output_dir = ROOT_PATH / "data" / f"input_{input_sr}Hz"
            os.makedirs(output_dir, exist_ok=True)

            for file_idx, input_path in enumerate(tqdm(selected_files, desc=reg_name)):
                save_path = (
                    str(output_dir / f"reconstructed_{reg_name}.wav")
                    if file_idx == 0
                    else None
                )

                try:
                    result, latencies, total_proc_time, audio_duration_sec = process_single_file(
                        input_path=input_path,
                        model=model,
                        bad_model=bad_model,
                        config=config,
                        device=device,
                        input_sr=input_sr,
                        output_sr=OUTPUT_SR,
                        chunk_size=chunk_size,
                        band_full=band_full,
                        version_name=reg_name,
                        use_ol=use_ol,
                        stream_cfg=stream_cfg,
                        save_path=save_path,
                    )
                except Exception as e:
                    print(f"[ERROR] file {input_path}: {e}")
                    continue

                all_latencies.extend(latencies)
                all_proc_times.append(total_proc_time)
                all_audio_durations.append(audio_duration_sec)

                for key in metrics_accum.keys():
                    if key in result and isinstance(result[key], (int, float)):
                        metrics_accum[key].append(float(result[key]))

            print(f"  Accumulated metric keys: {list(metrics_accum.keys())}")

            total_audio_sec = float(np.sum(all_audio_durations)) if all_audio_durations else 0.0
            total_proc_sec = float(np.sum(all_proc_times)) if all_proc_times else 0.0

            summary_data[input_sr][reg_name] = {
                "snr": _mean_or_zero(metrics_accum["snr"]),
                "lsd": _mean_or_zero(metrics_accum["lsd"]),
                "lsd_hf": _mean_or_zero(metrics_accum["lsd_hf"]),
                "lsd_lf": _mean_or_zero(metrics_accum["lsd_lf"]),
                "latency_ms": (_mean_or_zero(all_latencies) * 1000.0) if all_latencies else 0.0,
                "rtf": _mean_or_zero(metrics_accum["rtf"]) if metrics_accum["rtf"] else (
                    total_proc_sec / max(total_audio_sec, 1e-8)
                ),
                "throughput": (total_audio_sec / total_proc_sec) if total_proc_sec > 0 else 0.0,
            }

    # ===================== SUMMARY =====================
    print("\n" + "=" * 100)
    print("SUMMARY — first 5 samples | regimes: sequential_no_overlap / sequential_overlap_50pct")
    print("=" * 100)

    for sr in TEST_INPUT_SRS:
        print(f"\n  Input SR = {sr} Hz")
        print("  Metric                      sequential_no_overlap     sequential_overlap_50pct")
        print("  ------------------------------------------------------------------------------")

        d = summary_data[sr]
        no = d["sequential_no_overlap"]
        ol = d["sequential_overlap_50pct"]

        rows = [
            ("SNR  (dB)", "snr"),
            ("LSD", "lsd"),
            ("LSD-HF", "lsd_hf"),
            ("LSD-LF", "lsd_lf"),
            ("Latency (ms)", "latency_ms"),
            ("RTF", "rtf"),
            ("Throughput", "throughput"),
        ]

        for label, key in rows:
            print(f"  {label:<20} {no[key]:>28.4f} {ol[key]:>28.4f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
