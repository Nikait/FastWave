import pandas as pd
import math

class MetricTracker:
    """
    Class to aggregate metrics from many batches, computing mean and std.
    """

    def __init__(self, *keys, writer=None):
        """
        Args:
            *keys (list[str]): metric names
            writer: (unused here) hook for external logging
        """
        self.writer = writer
        # add 'sumsq' and 'std' columns
        cols = ["total", "counts", "average", "sumsq", "std"]
        self._data = pd.DataFrame(index=keys, columns=cols, dtype=float)
        self.reset()

    def reset(self):
        """Zero out all accumulators."""
        for col in self._data.columns:
            self._data[col].values[:] = 0.0

    def update(self, key, value, n=1):
        """
        Add `value` observed `n` times to metric `key`.
        Recompute mean and std in O(1).
        """
        # total and count
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        # sum of squares
        self._data.loc[key, "sumsq"] += (value ** 2) * n

        # recompute average
        total = self._data.loc[key, "total"]
        count = self._data.loc[key, "counts"]
        mean = total / count
        self._data.loc[key, "average"] = mean

        # recompute std = sqrt(E[x^2] - E[x]^2)
        sumsq = self._data.loc[key, "sumsq"]
        variance = sumsq / count - mean ** 2
        variance = max(0.0, variance)
        self._data.loc[key, "std"] = math.sqrt(variance)

    def avg(self, key):
        """Return the running mean for `key`."""
        return float(self._data.loc[key, "average"])

    def std(self, key):
        """Return the running standard deviation for `key`."""
        return float(self._data.loc[key, "std"])

    def result(self):
        """Get dict of all running means and stds."""
        # return a joint dict: key -> {mean, std}
        return {
            key: {
                "mean": float(self._data.loc[key, "average"]),
                "std": float(self._data.loc[key, "std"])
            }
            for key in self._data.index
        }

    def result_std(self):
        """Get dict of all running standard deviations."""
        return dict(self._data["std"])

    def keys(self):
        """List all tracked metric names."""
        return self._data.index
