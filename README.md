# FastWave: Optimized Diffusion Model for Audio Super-Resolution

## TL;DR
FastWave is a lightweight diffusion model for general audio super-resolution (any -> 48 kHz) that combines **[EDM](https://arxiv.org/abs/2206.00364)** with **[ConvNeXtV2](https://arxiv.org/abs/2301.00808)** architectural improvements. It matches SOTA quality with just **1.3 M parameters**, **~50 GFLOPs** total at 4 NFE or 8 for slightly better quality, and trains on a single GPU in a fraction of the time required by competing approaches.

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

FastWave builds on the NU-Wave 2 backbone with two independent sets of changes.

### From [EDM](https://arxiv.org/abs/2206.00364)

Instead of predicting noise $\epsilon$, FastWave is trained as an explicit denoiser $D_\theta(x+n;\sigma)\approx x$ with $\sigma$-based preconditioning ($c_\text{in}$, $c_\text{skip}$, $c_\text{out}$) that keeps input/output magnitudes stable throughout training. The noise level is drawn from a log-normal distribution whose parameters $P_\text{mean}$ and $P_\text{std}$ are estimated directly from the training data, concentrating learning on the most informative noise levels. At inference, a continuous EDM noise schedule replaces the fixed log-SNR schedule of NU-Wave 2, enabling high-quality reconstruction with as few as **4 NFE**.

### From [ConvNeXtV2](https://arxiv.org/abs/2301.00808)

Two targeted changes reduce model size from 1.8 M to **1.3 M parameters** and cut per-step FLOPs from 18.99 to **12.87 GFLOPs**. First, standard `Conv1d` layers in the FFC local branch and the BSFT shared MLP are replaced with **depthwise separable convolutions** (depthwise + pointwise), slashing parameter count while preserving the receptive field. Second, **Global Response Normalization (GRN)** is inserted after each depthwise transformation to restore cross-channel interaction that depthwise convolutions naturally limit.

The original general architecture is preserved as in **[NU-Wave 2](https://arxiv.org/abs/2206.08545)** Figure 1, mainly the architecture inside the STFC and BSFT blocks changes, we attach a picture:

<img width="2670" height="1562" alt="image" src="https://github.com/user-attachments/assets/727eee21-d564-4584-8293-c7026494d925" />
