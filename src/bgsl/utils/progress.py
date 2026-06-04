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
_SingleLineStream wraps stderr and intercepts tqdm's writes:
  - \\r-containing writes get \\r + ANSI clear-line (\\x1b[2K) so the bar
    stays on one physical line without producing new lines.
  - Plain \\n writes (epoch summaries, metric prints) pass through unchanged.

CloudProgressBar
----------------
Subclasses TQDMProgressBar and passes _SingleLineStream as the ``file=``
argument at tqdm construction time (the public API), avoiding any
post-construction patching of private attributes.

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
# _SingleLineStream
# ---------------------------------------------------------------------------

class _SingleLineStream:
    """
    ANSI-aware stream wrapper that keeps tqdm on a single visual line.

    tqdm emits ``\\r`` characters to overwrite the progress bar in-place.
    Non-interactive or piped environments turn those into literal text,
    producing thousands of identical-looking log lines.

    This wrapper:
      * Splits on ``\\r`` and emits ``\\r\\x1b[2K`` (move to col-0, clear line)
        before each segment so the display stays on one line.
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
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


# Module-level singleton so all bars share one stream instance.
_CLOUD_STREAM = _SingleLineStream(sys.stderr)


# ---------------------------------------------------------------------------
# CloudProgressBar
# ---------------------------------------------------------------------------

class CloudProgressBar(TQDMProgressBar):
    """
    PyTorch Lightning progress bar safe for cloud / headless environments.

    Overrides each ``init_*_tqdm`` method to inject ``file=_CLOUD_STREAM``
    at tqdm construction time.  The base class bar is closed cleanly and a
    new bar is created with identical configuration plus the safe stream.

    This approach uses the documented ``tqdm`` public API (``file=``) rather
    than patching the private ``bar.fp`` attribute after construction.
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _rewrap(bar: tqdm) -> tqdm:
        """
        Close *bar* and return a new tqdm instance with identical settings
        but with ``file=_CLOUD_STREAM``.

        All attributes read here (``total``, ``desc``, ``unit``, etc.) are
        part of tqdm's public interface and documented in tqdm's API.
        """
        bar_class = type(bar)
        kwargs: dict = dict(
            total=bar.total,
            desc=bar.desc,
            unit=bar.unit,
            dynamic_ncols=bar.dynamic_ncols,
            leave=bar.leave,
            miniters=bar.miniters,
            mininterval=10.0,  # Cloud-friendly logging interval (10s)
            bar_format=bar.bar_format,
            position=bar.pos,
            disable=bar.disable,
            file=_CLOUD_STREAM,
        )
        bar.close()
        return bar_class(**kwargs)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def init_sanity_tqdm(self) -> tqdm:
        return self._rewrap(super().init_sanity_tqdm())

    def init_train_tqdm(self) -> tqdm:
        return self._rewrap(super().init_train_tqdm())

    def init_validation_tqdm(self) -> tqdm:
        return self._rewrap(super().init_validation_tqdm())

    def init_test_tqdm(self) -> tqdm:
        return self._rewrap(super().init_test_tqdm())

    def init_predict_tqdm(self) -> tqdm:
        return self._rewrap(super().init_predict_tqdm())