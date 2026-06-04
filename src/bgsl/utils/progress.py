"""
bgsl/utils/progress.py
-----------------------
Cloud-safe progress bar for PyTorch Lightning training.

Problem
-------
PyTorch Lightning's default TQDMProgressBar uses tqdm, which writes ``\\r``
(carriage return) to overwrite the progress bar in-place. This works in
interactive terminals but fails in piped/cloud environments:

  subprocess → pipe → executor → Colab/Slurm/nohup

In these environments ``\\r`` does NOT cause visual line-overwriting. Each
tqdm refresh appears as a new line, flooding output with hundreds of
near-identical lines per epoch.

Solution
--------
_CloudStream intercepts tqdm's writes and applies two rules:

  1. ``\\r``-only writes (step-level progress updates) are **buffered** — not
     emitted immediately. A periodic heartbeat (default 30 s) emits the
     latest buffered state so the user can see training is alive.
  2. ``\\n`` writes (epoch end, bar close, metric summaries) **flush the
     buffer** and pass through immediately.

This produces ~2-4 clean lines per epoch instead of hundreds.

tqdm's write pattern
--------------------
tqdm's internal ``sp()`` always writes ``\\r`` + bar_text as a single call.
At bar close (``leave=True``), tqdm calls ``display()`` (which writes
``\\r`` + final_text) then **separately** writes ``\\n``.  So ``\\r`` and
``\\n`` are always separate ``write()`` calls — our detection is reliable.

YAML registration
-----------------
Add as the first callback entry:

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
import time
from typing import Optional

from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.std import tqdm


# ---------------------------------------------------------------------------
# _CloudStream
# ---------------------------------------------------------------------------

class _CloudStream:
    """
    Stream wrapper that produces clean, epoch-level output in cloud/piped
    environments.

    Strategy
    --------
    * ``\\r``-only writes → buffered, emitted at most once per heartbeat
      interval (default 30 s).
    * ``\\n``-containing writes → the last buffered ``\\r`` content is
      flushed first (as a ``\\n``-terminated line), then the ``\\n`` write
      itself is passed through.  Bare ``\\n`` (from tqdm's separate
      ``fp.write('\\n')`` at bar close) is suppressed since we already
      emitted the final bar state from the buffer.
    * Other writes → passed through unchanged.
    """

    def __init__(
        self,
        stream=None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._heartbeat_interval = heartbeat_interval
        self._last_emit_time: float = 0.0
        self._pending_cr: str = ""  # most recent \\r-only write

    def write(self, s: str) -> None:
        if not s:
            return

        # ----- Case 1: contains \\n → finalization (bar close / metric log)
        if "\n" in s:
            # Emit whatever was buffered from \\r writes — that's the final
            # bar state (e.g. "Epoch 0: 100%|██████| 1993/1993 [01:20...]").
            if self._pending_cr:
                clean = self._pending_cr.replace("\r", "")
                if clean.strip():
                    self._stream.write(clean + "\n")
                    self._stream.flush()
                self._pending_cr = ""

            # Now handle the \\n write itself.  tqdm's bar close writes a
            # bare "\\n" after the final display() — skip it since we
            # already emitted the bar content above.  Anything with real
            # text content (metric logs, summaries) passes through.
            clean_s = s.replace("\r", "")
            if clean_s.strip():
                self._stream.write(clean_s)
                self._stream.flush()
            return

        # ----- Case 2: \\r without \\n → step-level progress update
        if "\r" in s:
            self._pending_cr = s

            # Heartbeat: periodically emit so user knows training is alive
            now = time.monotonic()
            if now - self._last_emit_time >= self._heartbeat_interval:
                clean = s.replace("\r", "").strip()
                if clean:
                    self._stream.write(clean + "\n")
                    self._stream.flush()
                self._last_emit_time = now
            return

        # ----- Case 3: plain text (no \\r, no \\n) → pass through
        self._stream.write(s)
        self._stream.flush()

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        # Tell tqdm this is a terminal so it uses \\r (not \\n) for step
        # updates.  That lets us distinguish step updates from epoch-end
        # writes and apply the buffering strategy above.
        return True

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

    Patches each tqdm bar's file handle to ``_CLOUD_STREAM`` so that output
    is buffered and emitted cleanly (see ``_CloudStream`` docstring).
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_bar(bar: tqdm) -> tqdm:
        """
        Patch an existing tqdm bar to use the cloud stream.

        We directly set ``bar.fp`` instead of closing and recreating the
        bar (which would invalidate tqdm's internal position tracking and
        could silently suppress the bar).
        """
        bar.fp = _CLOUD_STREAM
        # Fixed width avoids dynamic terminal queries that fail in pipes.
        bar.dynamic_ncols = False
        bar.ncols = 120
        # 10 s refresh reduces the volume of writes into _CloudStream.
        # _CloudStream further throttles to 30 s heartbeats.
        bar.mininterval = 10.0
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