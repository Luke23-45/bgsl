"""
bgsl/utils/progress.py
-----------------------
Cloud-safe progress bar for PyTorch Lightning training.

Problem
-------
PyTorch Lightning's default TQDMProgressBar writes carriage returns (\\r) to
stderr. In headless cloud environments (Slurm, nohup, piped stdout) those \\r
characters either print as literal text (flooding logs) or stall the output
buffer, making the job look stuck after printing the model summary.

Solution
--------
CloudProgressBar subclasses TQDMProgressBar and overrides the init_*_tqdm
methods to directly patch the file handle on the tqdm bar to our cloud-safe
stream. This avoids the fragile close-and-recreate pattern that caused
tqdm position tracking issues (bar.pos becomes None after close, leading
to silent progress bars).

_CloudStream wraps stderr and converts \\r-based overwrites into \\n-terminated
lines so cloud loggers can display progress updates as they arrive.

YAML registration
-----------------
Add as the first callback entry so Lightning registers it before its own
progress-bar logic initialises:

    trainer:
      enable_progress_bar: true
      callbacks:
        - class_path: bgsl.utils.progress.CloudProgressBar
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          ...

Reference: plan.md §13.1 (standalone package, no APEX-MoE deps).
"""

from __future__ import annotations

import sys
from typing import Optional

from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.std import tqdm


# ---------------------------------------------------------------------------
# _CloudStream
# ---------------------------------------------------------------------------

class _CloudStream:
    """
    Stream wrapper that makes tqdm output cloud-friendly.

    tqdm emits ``\\r`` characters to overwrite the progress bar in-place.
    Non-interactive or piped environments turn those into literal text,
    producing thousands of identical-looking log lines.

    This wrapper:
      * Converts ``\\r``-containing writes to ``\\n``-terminated lines so
        cloud loggers (Colab, Slurm, nohup) actually flush and display them.
      * Passes ``\\n``-terminated writes (epoch logs) straight through.
      * Implements ``flush()``, ``isatty()``, and ``encoding`` so tqdm is fully
        satisfied without any monkey-patching.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stderr

    def write(self, s: str) -> None:
        if not s:
            return

        if "\r" in s:
            # Cloud loggers are often line-buffered and ignore \\r until \\n.
            # We strip \\r and enforce \\n so it actually appears in the logs.
            s = s.replace("\r", "")
            if not s.endswith("\n"):
                s += "\n"
            self._stream.write(s)
            self._stream.flush()
            return

        # Plain write (metric lines, newlines): pass through unchanged.
        self._stream.write(s)
        self._stream.flush()

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False  # Always report non-interactive for cloud safety

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


# Module-level singleton so all bars share one stream instance.
_CLOUD_STREAM = _CloudStream(sys.stderr)


# ---------------------------------------------------------------------------
# CloudProgressBar
# ---------------------------------------------------------------------------

class CloudProgressBar(TQDMProgressBar):
    """
    PyTorch Lightning progress bar safe for cloud / headless environments.

    Instead of closing and recreating tqdm bars (which breaks tqdm's internal
    position tracking and can silently disable bars), we directly patch the
    ``fp`` attribute on the existing bar to use our cloud-safe stream.

    We also lower ``mininterval`` to 2 seconds (from tqdm's default 0.1s)
    which balances log volume with responsiveness — enough to see progress
    without flooding cloud logs.
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_bar(bar: tqdm) -> tqdm:
        """
        Patch an existing tqdm bar to use the cloud stream.

        Instead of closing and recreating (which invalidates bar.pos and
        can silently suppress the bar), we directly set the file handle.
        This is safe because tqdm reads ``self.fp`` on every write.
        """
        bar.fp = _CLOUD_STREAM
        # Lower mininterval so progress shows up within a couple seconds
        # instead of the default 0.1s (too frequent for logs) or 10s (too slow).
        bar.mininterval = 2.0
        bar.miniters = 1
        return bar

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def init_sanity_tqdm(self) -> tqdm:
        return self._patch_bar(super().init_sanity_tqdm())

    def init_train_tqdm(self) -> tqdm:
        return self._patch_bar(super().init_train_tqdm())

    def init_validation_tqdm(self) -> tqdm:
        return self._patch_bar(super().init_validation_tqdm())

    def init_test_tqdm(self) -> tqdm:
        return self._patch_bar(super().init_test_tqdm())

    def init_predict_tqdm(self) -> tqdm:
        return self._patch_bar(super().init_predict_tqdm())