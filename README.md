# FastWave: Optimized Diffusion Model for Audio Super-Resolution

:scroll: [Arxiv](https://arxiv.org/abs/2603.04122)  &nbsp; 


## TL;DR
FastWave is a lightweight diffusion model for general audio super-resolution (any -> 48 kHz). It matches SOTA quality with just **1.3 M parameters**, **~50 GFLOPs** total at 4 NFE or 8 for slightly better quality, and trains on a single GPU in a fraction of the time required by competing approaches.

---

## Model Variants

We track three successive model versions:

|         Version          |                        Description                             |
|--------------------------|----------------------------------------------------------------|
| **NU-Wave 2** (Baseline) | Original model without modifications                           |
| **NU-Wave 2 + EDM**      | Baseline architecture retrained with EDM framework             |
| **FastWave**             | EDM diffusion modeling + ConvNeXtV2 architectural improvements |

---

## Architecture

FastWave builds on the [NU-Wave 2](https://arxiv.org/abs/2206.08545) backbone with two independent sets of changes.

### From [EDM](https://arxiv.org/abs/2206.00364)

Instead of predicting noise $\epsilon$, FastWave is trained as an explicit denoiser $D_\theta(x+n;\sigma)\approx x$ with $\sigma$-based preconditioning ($c_\text{in}$, $c_\text{skip}$, $c_\text{out}$) that keeps input/output magnitudes stable throughout training. The noise level is drawn from a log-normal distribution whose parameters $P_\text{mean}$ and $P_\text{std}$ are estimated directly from the training data, concentrating learning on the most informative noise levels. At inference, a continuous EDM noise schedule replaces the fixed log-SNR schedule of NU-Wave 2, enabling high-quality reconstruction with as few as 4 NFE.

### From [ConvNeXtV2](https://arxiv.org/abs/2301.00808)

Two targeted changes reduce model size from 1.8 M to **1.3 M parameters** and cut per-step FLOPs from 18.99 to **12.87 GFLOPs**. First, standard `Conv1d` layers in the FFC local branch and the BSFT shared MLP are replaced with **depthwise separable convolutions** (depthwise + pointwise), slashing parameter count while preserving the receptive field. Second, **Global Response Normalization (GRN)** is inserted after each depthwise transformation to restore cross-channel interaction that depthwise convolutions naturally limit.

The original general architecture is preserved as in NU-Wave 2 Figure 1, mainly the architecture inside the STFC and BSFT blocks changes, we attach a picture:

<img width="2670" height="1562" alt="image" src="https://github.com/user-attachments/assets/727eee21-d564-4584-8293-c7026494d925" />

### Comparison with Pretrained / Large-Capacity Models


## Results

All models are evaluated on the **VCTK dataset** (48 kHz, 8 speakers test set), measuring upsampling from 8 / 12 / 16 / 24 kHz to 48 kHz. 
FLOPs are given for one function evaluation.
For comparison we include [AudioSR](https://arxiv.org/abs/2309.07314) (latent diffusion model) and [FlowHigh](https://arxiv.org/abs/2501.04926) (single-step conditional flow matching, ICASSP 2025) as strong external baselines.

| Metric | FastWave 4 NFE | FastWave 8 NFE | NU-Wave 2 8 NFE | FlowHigh | AudioSR |
|---|:---:|:---:|:---:|:---:|:---:|
| **8 kHz** | | | | | |
| SNR ↑ | **18.75 ± 4.84** | 18.53 ± 4.73 | 18.43 ± 4.92 | 18.04 ± 4.74 | 13.75 ± 3.83 |
| LSD ↓ | 1.18 ± 0.12 | 1.19 ± 0.11 | 1.15 ± 0.10 | **0.96 ± 0.08** | 1.55 ± 0.15 |
| LSD-LF ↓ | 0.36 ± 0.08 | 0.28 ± 0.05 | 0.22 ± 0.07 | **0.24 ± 0.02** | 0.44 ± 0.07 |
| LSD-HF ↓ | 1.27 ± 0.13 | 1.29 ± 0.12 | 1.25 ± 0.11 | **1.05 ± 0.09** | 1.69 ± 0.17 |
| **12 kHz** | | | | | |
| SNR ↑ | 21.08 ± 5.71 | 20.93 ± 5.80 | 20.95 ± 5.18 | **21.17 ± 5.39** | 16.18 ± 3.96 |
| LSD ↓ | 1.09 ± 0.11 | 1.06 ± 0.09 | 1.02 ± 0.08 | **0.90 ± 0.09** | 1.46 ± 0.16 |
| LSD-LF ↓ | 0.49 ± 0.10 | 0.38 ± 0.06 | 0.27 ± 0.07 | **0.28 ± 0.05** | 0.55 ± 0.13 |
| LSD-HF ↓ | 1.21 ± 0.13 | 1.20 ± 0.11 | 1.16 ± 0.09 | **1.03 ± 0.10** | 1.65 ± 0.18 |
| **16 kHz** | | | | | |
| SNR ↑ | 23.07 ± 5.85 | 23.08 ± 6.06 | 23.31 ± 5.17 | **23.58 ± 5.41** | 19.25 ± 3.82 |
| LSD ↓ | 1.04 ± 0.10 | 0.98 ± 0.08 | 0.94 ± 0.08 | **0.85 ± 0.09** | 1.37 ± 0.15 |
| LSD-LF ↓ | 0.59 ± 0.13 | 0.44 ± 0.08 | 0.30 ± 0.09 | **0.28 ± 0.05** | 0.54 ± 0.13 |
| LSD-HF ↓ | 1.17 ± 0.12 | 1.14 ± 0.10 | 1.12 ± 0.09 | **1.02 ± 0.11** | 1.63 ± 0.18 |
| **24 kHz** | | | | | |
| SNR ↑ | 27.09 ± 4.84 | 27.22 ± 5.33 | 27.68 ± 4.21 | **27.80 ± 4.95** | 23.03 ± 3.48 |
| LSD ↓ | 0.93 ± 0.08 | 0.83 ± 0.06 | 0.78 ± 0.06 | **0.74 ± 0.09** | 1.27 ± 0.15 |
| LSD-LF ↓ | 0.66 ± 0.14 | 0.48 ± 0.09 | 0.33 ± 0.11 | **0.30 ± 0.06** | 0.58 ± 0.15 |
| LSD-HF ↓ | 1.08 ± 0.10 | 1.05 ± 0.09 | 1.04 ± 0.08 | **1.00 ± 0.13** | 1.69 ± 0.22 |
| **Complexity** | | | | | |
| RTF ↓ | 0.16 ± 0.03 | 0.30 ± 0.14 | 0.26 ± 0.02 | **0.06 ± 0.02** | 4.99 ± 1.59 |
| GFLOPs ↓ | **12.87** | **12.87** | 18.99 | 30.39 | 2536.2 |
| #params ↓ | **1.3 M** | **1.3 M** | 1.8 M | 49.4 M | 1285.4 M |


## Setup

### Requirements

```bash
conda create -n fastwave python=3.11 -y
conda activate fastwave

git clone https://github.com/Nikait/FastWave.git
cd FastWave
pip install -r requirements.txt
```


## Dataset

### Dataset

FastWave was trained and evaluated on the **VCTK dataset** (48 kHz, ~44 hours of speech from 110 speakers).

1. Download VCTK from the [official source](https://datashare.ed.ac.uk/handle/10283/3443).
2. Remove speaker p280 and p315
3. Run flac2wav.py on this dataset
4. As the output you should have wav48_silence_trimmed_wav directory with dataset next to the train.py file


## Configuration

All configuration files are located in `src/configs/`.
```
src/configs/
├── baseline.yaml       # Main training config
├── inference.yaml      # Inference config
├── dataloader/
├── datasets/
├── metrics/
├── model/
├── transforms/
└── writer/
```

### Training

Before training, configure `src/configs/baseline.yaml`. The project uses **Comet ML** for experiment tracking — paste your API key in the writer config section.

### Inference

Before running inference, configure `src/configs/inference.yaml` with:
- Path to the model checkpoint
- Input sample rate of your audio

To run / reproduce our checkpoint, firstly, download it from [click](https://drive.google.com/file/d/1oNCxrKjgiWsYGW6P49rsI84vFYR5G3m8/view?usp=sharing)
 then place to the saved/edm_convnetxt directory in the project directory.

## Training

```bash
python train.py
```

## Inference

```bash
python inference.py
```


## Citation

If you would like to cite us, please consider following form:

```bib
@misc{kuznetsov2026fastwaveoptimizeddiffusionmodel,
      title={FastWave: Optimized Diffusion Model for Audio Super-Resolution}, 
      author={Nikita Kuznetsov and Maksim Kaledin},
      year={2026},
      eprint={2603.04122},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.04122}, 
}
```
