import os
import comet_ml
from datetime import datetime
import numpy as np
import pandas as pd

class CometMLWriter:
    """
    Wraps a comet_ml.Experiment to mirror common logging methods.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initialize CometMLWriter using args/kwargs from Hydra.
        Expects kwargs: api_key, project_name, workspace, log_checkpoints.
        Optional: api_key can be omitted if COMET_API_KEY env var is set.
        Any extra kwargs are forwarded to comet_ml.Experiment.
        """
        # Extract known parameters
        api_key = kwargs.pop('api_key', None) or os.environ.get('COMET_API_KEY')
        project_name = kwargs.pop('project_name', None)
        print("PROJECT NAME:", project_name)
        workspace = kwargs.pop('workspace', None)
        log_checkpoints = kwargs.pop('log_checkpoints', False)

        if api_key is None:
            raise ValueError(
                'CometMLWriter requires an API key via `api_key` kwarg or COMET_API_KEY env variable.'
            )

        # Instantiate Comet experiment
        self.experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            **kwargs,
        )

        # Legacy alias
        self.exp = self.experiment
        self.log_checkpoints = log_checkpoints
        self.step = None

    def set_step(self, step: int, mode: str = None):
        """
        Set the current logging step.
        """
        self.step = step
        try:
            self.experiment.set_step(step)
        except Exception:
            pass

    def add_scalar(self, name: str, value: float, step: int = None):
        """
        Log a single scalar metric.
        """
        _step = step if step is not None else self.step
        self.experiment.log_metric(name, value, step=_step)

    def add_scalars(self, scalars: dict, step: int = None):
        """
        Log multiple scalar metrics at once.
        """
        _step = step if step is not None else self.step
        for name, val in scalars.items():
            self.experiment.log_metric(name, val, step=_step)

    def add_image(self, name: str, image, step: int = None):
        _step = step if step is not None else self.step
        # Use comet log_figure for matplotlib figures or arrays
        try:
            self.experiment.log_figure(figure=image, figure_name=name, step=_step)
        except Exception:
            # fallback to log_image without args
            self.experiment.log_image(image)

    def add_audio(self, name: str, audio, sample_rate: int, step: int = None):
        """
        Log an audio clip (numpy array or torch Tensor).
        """
        _step = step if step is not None else self.step
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()
        # Comet expects `file_name`
        self.experiment.log_audio(audio_data=audio, sample_rate=sample_rate, file_name=name, step=_step)

    def add_text(self, name: str, text: str, step: int = None):
        """
        Log a text string.
        """
        _step = step if step is not None else self.step
        self.experiment.log_text(text=text, step=_step)

    def add_histogram(self, name: str, values, bins: int = None, step: int = None):
        """
        Log a histogram of values (numpy or torch tensor).
        """
        _step = step if step is not None else self.step
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        self.experiment.log_histogram_3d(values=values, name=name, step=_step)

    def add_table(self, name: str, table: pd.DataFrame, step: int = None):
        """
        Log a pandas DataFrame as a table.
        """
        _step = step if step is not None else self.step
        # Comet doesn't accept step in log_table; set it manually
        self.experiment.set_step(_step)
        self.experiment.log_table(
            filename=f"{name}.csv",
            tabular_data=table,
            headers=True,
        )

    def add_checkpoint(self, local_path: str, artifact_path: str, step: int = None):
        """
        Log a model checkpoint file.
        """
        _step = step if step is not None else self.step
        self.experiment.set_step(_step)
        self.experiment.log_asset(local_path, artifact_path)