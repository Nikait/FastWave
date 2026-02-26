import math
import numpy as np


BASE_SCHEDULE_8 = np.array([-2.6, -0.8, 2.0, 6.4, 9.8, 12.9, 14.4, 17.2], dtype=float)

def make_schedule(
    num_steps: int,
    lambda_start: float = 20.0,
    lambda_end:   float = -20.0,
    base_schedule: np.ndarray = BASE_SCHEDULE_8,
) -> np.ndarray:
    """
    Return a log-SNR schedule of length `num_steps`.
    
    - If num_steps == len(base_schedule), returns base_schedule exactly.
    - Otherwise, inverts base_schedule -> t, linearly resamples in t, then reapplies lambda(t).
    
    Args:
      num_steps:     desired number of inference steps N.
      lambda_start:  lambda_0 endpoint (default 20).
      lambda_end:    lambda_1 endpoint (default −20).
      base_schedule: hand-tuned lambda’s for N=8.
    Returns:
      Array of length N with a valid DDIM/NU-Wave 2 log-SNR curve.
    """

    b = math.atan(math.exp(-lambda_start / 2.0))
    a = math.atan(math.exp(-lambda_end   / 2.0)) - b

    # exact eight-step case: return hand-tuned
    if num_steps == base_schedule.shape[0]:
        return base_schedule.copy()

    t_base = (np.arctan(np.exp(-base_schedule / 2.0)) - b) / a

    t_new = np.linspace(t_base[0], t_base[-1], num_steps)

    lambda_new = -2.0 * np.log(np.tan(a * t_new + b))

    return lambda_new

