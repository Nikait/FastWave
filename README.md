# FastWave: Optimized Diffusion Model for Audio Super-Resolution

## TL;DR
FastWave is a lightweight diffusion model for general audio super-resolution (any -> 48 kHz) that combines **[EDM training](https://arxiv.org/abs/2206.00364)** with **[ConvNeXtV2](https://arxiv.org/abs/2301.00808)** architectural improvements. It matches SOTA quality with just **1.3 M parameters**, **~50 GFLOPs** total at 4 NFE or 8 for slightly better quality, and trains on a single GPU in a fraction of the time required by competing approaches.

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

